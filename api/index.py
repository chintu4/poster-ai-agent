import asyncio
import base64
import html
import json
import os
import re
import time
from typing import Literal, Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

WORKER_URL = os.getenv(
    "HTML_TO_PNG_WORKER_URL",
    "https://html-to-png-worker.chintusharma00014.workers.dev/",
)
SUMMARY_CHAR_THRESHOLD = 2000
MAX_RAW_CHARS = 12000

app = FastAPI(title="Poster AI Agent v2 (Agno + Groq)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    brief: str = Field(..., min_length=5, max_length=20000)
    template: Optional[Literal["dark-tech", "clean-light", "bold-gradient"]] = None
    image_data_url: Optional[str] = Field(default=None, max_length=4_000_000)
    include_stock_image: bool = True
    render_png: bool = True


class MetricItem(BaseModel):
    label: str
    value: str


class PosterSpec(BaseModel):
    template: Literal["dark-tech", "clean-light", "bold-gradient"] = "dark-tech"
    badge: str
    headline: str
    subhead: str
    bullets: list[str] = Field(..., min_length=2, max_length=4)
    icon_queries: list[str] = Field(default_factory=lambda: ["zap", "shield-check", "rocket"])
    metrics: list[MetricItem] = Field(default_factory=list)
    image_query: Optional[str] = None
    cta: str
    footer_note: str = "chintu-work.vercel.app"
    accent_hex: str = "#3b82f6"


class TraceStep(BaseModel):
    step: str
    status: Literal["completed", "fallback", "skipped"]
    duration_ms: int
    detail: str


class GenerateResponse(BaseModel):
    spec: PosterSpec
    html: str
    png_data_url: Optional[str] = None
    summarized: bool = False
    original_chars: int
    effective_chars: int
    mode: Literal["live", "fallback"]
    trace: list[TraceStep]


def _extract_json_block(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No valid JSON object found in model output")


def _clamp_spec(raw: dict, preferred_template: Optional[str]) -> PosterSpec:
    template = preferred_template or raw.get("template") or "dark-tech"
    if template not in ("dark-tech", "clean-light", "bold-gradient"):
        template = "dark-tech"

    bullets = [str(b).strip()[:90] for b in (raw.get("bullets") or []) if str(b).strip()][:3]
    while len(bullets) < 3:
        bullets.append("Production-ready AI workflow & deterministic validation")

    raw_metrics = raw.get("metrics") or []
    metrics: list[MetricItem] = []
    for m in raw_metrics[:3]:
        if isinstance(m, dict) and m.get("label") and m.get("value"):
            metrics.append(
                MetricItem(
                    label=str(m["label"]).strip()[:24],
                    value=str(m["value"]).strip()[:14],
                )
            )

    icon_queries = [
        re.sub(r"[^a-z0-9-]", "", str(q).lower().strip().replace(" ", "-")) or "sparkles"
        for q in (raw.get("icon_queries") or ["zap", "shield-check", "rocket"])[:3]
    ]
    while len(icon_queries) < 3:
        icon_queries.append("check-circle")

    accent = str(raw.get("accent_hex") or "#3b82f6").strip()
    if not re.match(r"^#[0-9a-fA-F]{6}$", accent):
        accent = "#3b82f6"

    return PosterSpec(
        template=template,
        badge=str(raw.get("badge") or "AI AGENT SHOWCASE").strip()[:36],
        headline=str(raw.get("headline") or "Build Smarter AI Workflows").strip()[:78],
        subhead=str(raw.get("subhead") or "Automated, observable, and ready for production.").strip()[:155],
        bullets=bullets,
        icon_queries=icon_queries,
        metrics=metrics,
        image_query=str(raw.get("image_query") or "").strip()[:60] or None,
        cta=str(raw.get("cta") or "Book a 15-min Call").strip()[:36],
        footer_note=str(raw.get("footer_note") or "chintu-work.vercel.app").strip()[:45],
        accent_hex=accent,
    )


def _deterministic_condense(text: str, limit: int = 1800) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rsplit(" ", 1)[0] + "..."


def _condense_with_agno(raw_brief: str) -> tuple[str, bool, str]:
    capped = raw_brief[:MAX_RAW_CHARS]
    if len(capped) <= SUMMARY_CHAR_THRESHOLD:
        return capped, False, "Within Groq token budget; passed directly"

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return (
            _deterministic_condense(capped),
            True,
            "GROQ_API_KEY missing; used deterministic condensation",
        )

    try:
        from agno.agent import Agent
        from agno.models.groq import Groq

        summarizer = Agent(
            name="Brief Summarizer",
            model=Groq(id="openai/gpt-oss-20b", max_tokens=280),
            instructions=[
                "Condense the user's long product/event brief into at most 150 words.",
                "Preserve: target audience, core value proposition, key numbers/stats, 3 benefits, and CTA.",
                "Output plain text only—no markdown headers.",
            ],
        )
        res = summarizer.run(
            f"Condense this brief for a 1080x1350 infographic poster:\n\n{capped}"
        )
        summary = str(getattr(res, "content", "") or "").strip()
        if summary and len(summary) < len(capped):
            return (
                summary,
                True,
                f"Summarized {len(raw_brief)} chars -> {len(summary)} chars via gpt-oss-20b",
            )
    except Exception as exc:
        condensed = _deterministic_condense(capped)
        return (
            condensed,
            True,
            f"Summarizer fallback ({type(exc).__name__}); condensed to {len(condensed)} chars",
        )

    condensed = _deterministic_condense(capped)
    return condensed, True, f"Condensed to {len(condensed)} chars"


async def _analyze_image_with_groq_vision(image_data_url: str) -> tuple[str, str]:
    """Use Groq's qwen/qwen3.8-27b vision model to inspect an uploaded product/brand image."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "", "GROQ_API_KEY missing; skipped vision analysis"

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)
        resp = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            max_tokens=180,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this product/brand image in 1-2 short sentences for a marketing poster, "
                                "and suggest one matching hex accent color (e.g. #3b82f6)."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, "Analyzed uploaded image via Groq Vision (qwen/qwen3.8-27b)"
    except Exception as exc:
        return "", f"Groq Vision fallback ({type(exc).__name__}); continued without vision context"


async def _fetch_single_iconify_svg(client: httpx.AsyncClient, query: str, accent: str) -> Optional[str]:
    try:
        search_resp = await client.get(
            "https://api.iconify.design/search",
            params={"query": query, "prefix": "lucide", "limit": 1},
        )
        icons = search_resp.json().get("icons") or []
        icon_id = icons[0] if icons else f"lucide:{query}"
        if ":" not in icon_id:
            icon_id = f"lucide:{icon_id}"
        prefix, name = icon_id.split(":", 1)
        svg_resp = await client.get(f"https://api.iconify.design/{prefix}/{name}.svg", params={"color": accent, "width": "28", "height": "28"})
        if svg_resp.status_code == 200 and "<svg" in svg_resp.text:
            return svg_resp.text
    except Exception:
        pass
    return None


async def _fetch_iconify_icons(queries: list[str], accent: str) -> tuple[list[Optional[str]], str]:
    async with httpx.AsyncClient(timeout=3.5) as client:
        svgs = await asyncio.gather(*[_fetch_single_iconify_svg(client, q, accent) for q in queries[:3]])
    found = sum(1 for s in svgs if s)
    return list(svgs), f"Fetched {found}/{len(queries[:3])} copyright-free Lucide SVGs from Iconify API"


async def _fetch_wikimedia_image(query: str) -> tuple[Optional[str], Optional[str], str]:
    """Fetch a CC/public-domain background photo from Wikimedia Commons (no API key required)."""
    try:
        async with httpx.AsyncClient(timeout=4.0, headers={"User-Agent": "PosterAIAgent/2.0"}) as client:
            resp = await client.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{query} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 3,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "iiurlwidth": 1080,
                    "format": "json",
                },
            )
            pages = (resp.json().get("query") or {}).get("pages") or {}
            for page in pages.values():
                info = (page.get("imageinfo") or [{}])[0]
                thumb = info.get("thumburl") or info.get("url")
                if thumb and any(thumb.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    return thumb, "Wikimedia Commons (CC)", f"Matched copyright-free photo for '{query}'"
        return None, None, f"No Wikimedia bitmap match for '{query}'; using clean typographic layout"
    except Exception as exc:
        return None, None, f"Stock image lookup skipped ({type(exc).__name__})"


def _fallback_poster_spec(brief: str, preferred_template: Optional[str]) -> PosterSpec:
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", brief) if len(s.strip()) > 6]
    pct_match = re.search(r"(\d+%)", brief)
    metrics = [MetricItem(label="Impact", value=pct_match.group(1))] if pct_match else []
    return _clamp_spec(
        {
            "template": preferred_template or "dark-tech",
            "badge": "AI AGENT SHOWCASE",
            "headline": sentences[0] if sentences else "Launch Faster with Autonomous AI Agents",
            "subhead": sentences[1] if len(sentences) > 1 else "Production-ready workflows built with Agno, Groq, and clean APIs.",
            "bullets": sentences[2:5],
            "icon_queries": ["zap", "shield-check", "rocket"],
            "metrics": [m.model_dump() for m in metrics],
            "image_query": "technology abstract dark",
            "cta": "Book a 15-min Call",
            "footer_note": "chintu-work.vercel.app",
            "accent_hex": "#3b82f6",
        },
        preferred_template,
    )


def render_poster_html(
    spec: PosterSpec,
    icon_svgs: list[Optional[str]],
    hero_image_url: Optional[str],
    image_credit: Optional[str],
) -> str:
    esc = html.escape
    accent = spec.accent_hex

    if spec.template == "clean-light":
        bg = "#ffffff"
        fg = "#09090b"
        muted = "#52525b"
        card_bg = "#f4f4f5"
        border = "#e4e4e7"
        scrim = "rgba(255,255,255,0.88)"
    elif spec.template == "bold-gradient":
        bg = f"linear-gradient(145deg, #0f172a 0%, #1e1b4b 55%, {accent}33 100%)"
        fg = "#f8fafc"
        muted = "#cbd5e1"
        card_bg = "rgba(255,255,255,0.08)"
        border = "rgba(255,255,255,0.16)"
        scrim = "rgba(15,23,42,0.82)"
    else:
        bg = "#09090b"
        fg = "#fafafa"
        muted = "#a1a1aa"
        card_bg = "#18181b"
        border = "#27272a"
        scrim = "rgba(9,9,11,0.84)"

    bg_layer = ""
    if hero_image_url and not hero_image_url.startswith("data:image/png;base64,"):
        bg_layer = f"""
        <div style="position:absolute;inset:0;background-image:url('{esc(hero_image_url)}');background-size:cover;background-position:center;opacity:0.22;z-index:0;"></div>
        <div style="position:absolute;inset:0;background:{scrim};z-index:0;"></div>
        """

    product_img_html = ""
    if hero_image_url and hero_image_url.startswith("data:image/"):
        product_img_html = f"""
        <div style="width:220px;height:220px;border-radius:24px;background:{card_bg};border:1px solid {border};display:flex;align-items:center;justify-content:center;padding:16px;flex-shrink:0;">
          <img src="{hero_image_url}" alt="Product visual" style="max-width:100%;max-height:100%;object-fit:contain;" />
        </div>
        """

    metrics_html = ""
    if spec.metrics:
        metric_cards = "".join(
            f"""
            <div style="flex:1;padding:20px 24px;border-radius:18px;background:{card_bg};border:1px solid {border};">
              <div style="font-size:40px;font-weight:800;color:{accent};line-height:1.1;">{esc(m.value)}</div>
              <div style="font-size:20px;color:{muted};margin-top:6px;font-weight:600;">{esc(m.label)}</div>
            </div>
            """
            for m in spec.metrics[:3]
        )
        metrics_html = f'<div style="display:flex;gap:16px;margin-top:28px;">{metric_cards}</div>'

    bullet_html = "\n".join(
        f"""
        <div style="display:flex;align-items:center;gap:18px;padding:20px 24px;border-radius:20px;background:{card_bg};border:1px solid {border};">
          <div style="width:46px;height:46px;border-radius:14px;background:{accent}18;border:1px solid {accent}44;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            {icon_svgs[i] if i < len(icon_svgs) and icon_svgs[i] else f'<div style="width:12px;height:12px;border-radius:999px;background:{accent};"></div>'}
          </div>
          <div style="font-size:26px;line-height:1.35;font-weight:500;color:{fg};">{esc(b)}</div>
        </div>
        """
        for i, b in enumerate(spec.bullets)
    )

    credit_html = f'<span style="font-size:16px;color:{muted};display:block;margin-top:4px;">Photo: {esc(image_credit)}</span>' if image_credit else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      position: relative;
      width: 1080px;
      height: 1350px;
      background: {bg};
      color: {fg};
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      padding: 76px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      overflow: hidden;
    }}
    .content {{ position: relative; z-index: 1; display: flex; flex-direction: column; justify-content: space-between; height: 100%; }}
  </style>
</head>
<body>
  {bg_layer}
  <div class="content">
    <div>
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:28px;">
        <div style="flex:1;">
          <div style="display:inline-block;padding:10px 22px;border-radius:999px;background:{accent}22;border:1px solid {accent}66;color:{accent};font-size:20px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:28px;">
            {esc(spec.badge)}
          </div>
          <h1 style="font-size:66px;line-height:1.08;font-weight:800;letter-spacing:-0.03em;margin-bottom:22px;">
            {esc(spec.headline)}
          </h1>
          <p style="font-size:28px;line-height:1.42;color:{muted};max-width:860px;">
            {esc(spec.subhead)}
          </p>
        </div>
        {product_img_html}
      </div>
      {metrics_html}
    </div>

    <div style="display:flex;flex-direction:column;gap:16px;">
      {bullet_html}
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;gap:24px;padding-top:30px;border-top:1px solid {border};">
      <div style="padding:18px 32px;border-radius:16px;background:{accent};color:#ffffff;font-size:24px;font-weight:700;white-space:nowrap;">
        {esc(spec.cta)} &rarr;
      </div>
      <div style="text-align:right;font-size:22px;color:{muted};font-weight:600;">
        {esc(spec.footer_note)}
        {credit_html}
      </div>
    </div>
  </div>
</body>
</html>"""


async def render_poster_png(html_content: str) -> tuple[Optional[str], str]:
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                WORKER_URL,
                json={"html": html_content, "width": 1080, "height": 1350},
            )
            content_type = resp.headers.get("content-type", "")
            if resp.status_code == 200 and "image/png" in content_type:
                b64 = base64.b64encode(resp.content).decode("ascii")
                return f"data:image/png;base64,{b64}", "Rendered 1080x1350 PNG via Cloudflare Worker"
            return None, f"Worker returned HTTP {resp.status_code}; showing live HTML preview"
    except Exception as exc:
        return None, f"Worker timeout/fallback ({type(exc).__name__}); showing live HTML preview"


@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "summary_char_threshold": SUMMARY_CHAR_THRESHOLD,
    }


@app.post("/api/generate", response_model=GenerateResponse)
@app.post("/generate", response_model=GenerateResponse)
@app.post("/", response_model=GenerateResponse)
async def generate_poster(req: GenerateRequest):
    trace: list[TraceStep] = []
    original_chars = len(req.brief)

    # Step 1: Summarization Guard
    t0 = time.perf_counter()
    effective_brief, summarized, summary_detail = _condense_with_agno(req.brief)
    trace.append(
        TraceStep(
            step="summarization_guard",
            status="completed" if summarized else "skipped",
            duration_ms=int((time.perf_counter() - t0) * 1000),
            detail=summary_detail,
        )
    )

    # Step 2: Optional Groq Vision inspection if user supplied an image
    vision_context = ""
    if req.image_data_url:
        tv = time.perf_counter()
        vision_context, vision_detail = await _analyze_image_with_groq_vision(req.image_data_url)
        trace.append(
            TraceStep(
                step="groq_vision_analysis",
                status="completed" if vision_context else "fallback",
                duration_ms=int((time.perf_counter() - tv) * 1000),
                detail=vision_detail,
            )
        )

    # Step 3: Agno Poster Agent -> PosterSpec
    t1 = time.perf_counter()
    mode: Literal["live", "fallback"] = "live"
    spec: Optional[PosterSpec] = None
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if api_key:
        try:
            from agno.agent import Agent
            from agno.models.groq import Groq

            poster_agent = Agent(
                name="Poster Copy & Infographic Agent",
                model=Groq(id="openai/gpt-oss-120b", max_tokens=750),
                instructions=[
                    "You design high-converting 1080x1350 marketing posters. Output ONLY a valid JSON object.",
                    "Keys: template ('dark-tech', 'clean-light', or 'bold-gradient'), badge (<=30 chars), headline (<=70 chars), subhead (<=140 chars), bullets (array of 3 concise items <=80 chars each), icon_queries (3 Lucide icon names like 'zap', 'shield-check', 'rocket'), metrics (array of up to 2 {label, value} KPI objects extracted from the brief, or empty), image_query (2-3 word stock photo search phrase), cta (<=30 chars), footer_note, accent_hex.",
                ],
            )
            prompt = f"Preferred template: {req.template or 'auto'}\n\nBrief:\n{effective_brief}"
            if vision_context:
                prompt += f"\n\nUploaded product image analysis:\n{vision_context}"

            result = poster_agent.run(prompt)
            raw_content = getattr(result, "content", None)
            if isinstance(raw_content, dict):
                spec = _clamp_spec(raw_content, req.template)
            else:
                spec = _clamp_spec(_extract_json_block(str(raw_content or "")), req.template)

            trace.append(
                TraceStep(
                    step="agno_poster_agent",
                    status="completed",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    detail="Generated & clamped PosterSpec via Agno + Groq (gpt-oss-120b)",
                )
            )
        except Exception as exc:
            mode = "fallback"
            spec = _fallback_poster_spec(effective_brief, req.template)
            trace.append(
                TraceStep(
                    step="agno_poster_agent",
                    status="fallback",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    detail=f"Groq/validation fallback ({type(exc).__name__}); used deterministic PosterSpec",
                )
            )
    else:
        mode = "fallback"
        spec = _fallback_poster_spec(effective_brief, req.template)
        trace.append(
            TraceStep(
                step="agno_poster_agent",
                status="fallback",
                duration_ms=int((time.perf_counter() - t1) * 1000),
                detail="GROQ_API_KEY not set; used deterministic PosterSpec",
            )
        )

    assert spec is not None

    # Step 4: Fetch copyright-free Iconify icons + Wikimedia stock photo in parallel
    ta = time.perf_counter()
    hero_image_url = req.image_data_url
    image_credit: Optional[str] = None
    if not hero_image_url and req.include_stock_image and spec.image_query:
        (icon_svgs, icon_detail), (hero_image_url, image_credit, img_detail) = await asyncio.gather(
            _fetch_iconify_icons(spec.icon_queries, spec.accent_hex),
            _fetch_wikimedia_image(spec.image_query),
        )
        asset_detail = f"{icon_detail}; {img_detail}"
    else:
        icon_svgs, asset_detail = await _fetch_iconify_icons(spec.icon_queries, spec.accent_hex)

    trace.append(
        TraceStep(
            step="copyright_free_assets",
            status="completed",
            duration_ms=int((time.perf_counter() - ta) * 1000),
            detail=asset_detail,
        )
    )

    # Step 5: Render deterministic HTML
    t2 = time.perf_counter()
    poster_html = render_poster_html(spec, icon_svgs, hero_image_url, image_credit)
    trace.append(
        TraceStep(
            step="html_template_render",
            status="completed",
            duration_ms=int((time.perf_counter() - t2) * 1000),
            detail=f"Built sanitized 1080x1350 '{spec.template}' HTML with icons & KPI strip",
        )
    )

    # Step 6: Render PNG via Cloudflare Worker
    png_data_url: Optional[str] = None
    if req.render_png:
        t3 = time.perf_counter()
        png_data_url, worker_detail = await render_poster_png(poster_html)
        trace.append(
            TraceStep(
                step="cloudflare_worker_png",
                status="completed" if png_data_url else "fallback",
                duration_ms=int((time.perf_counter() - t3) * 1000),
                detail=worker_detail,
            )
        )

    return GenerateResponse(
        spec=spec,
        html=poster_html,
        png_data_url=png_data_url,
        summarized=summarized,
        original_chars=original_chars,
        effective_chars=len(effective_brief),
        mode=mode,
        trace=trace,
    )
