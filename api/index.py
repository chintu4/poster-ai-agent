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

app = FastAPI(title="Poster AI Agent v2.2 (Agno + Groq + Infographics)")
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


class InfographicItem(BaseModel):
    label: str
    value: str
    percent: int = Field(default=75, ge=5, le=100)


class PosterSpec(BaseModel):
    template: Literal["dark-tech", "clean-light", "bold-gradient"] = "dark-tech"
    badge: str
    headline: str
    subhead: str
    bullets: list[str] = Field(..., min_length=2, max_length=4)
    icon_queries: list[str] = Field(default_factory=lambda: ["zap", "shield-check", "rocket"])
    metrics: list[MetricItem] = Field(default_factory=list)
    infographic_type: Literal["bars", "donut", "steps"] = "bars"
    infographic_title: str = "Key Impact"
    infographic_items: list[InfographicItem] = Field(default_factory=list)
    image_query: Optional[str] = None
    cta: str
    footer_note: str = ""
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


def _word_trim(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    trimmed = cleaned[: limit - 1].rsplit(" ", 1)[0]
    return (trimmed if len(trimmed) > limit // 2 else cleaned[: limit - 1]).rstrip(".,;:-") + "…"


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


def _clamp_spec(raw: dict, preferred_template: Optional[str] = None) -> PosterSpec:
    template = preferred_template or raw.get("template") or "dark-tech"
    if template not in ("dark-tech", "clean-light", "bold-gradient"):
        template = "dark-tech"

    bullets = [_word_trim(str(b), 78) for b in (raw.get("bullets") or []) if str(b).strip()][:3]
    while len(bullets) < 3:
        bullets.append("Built for speed, reliability, and measurable results")

    raw_metrics = raw.get("metrics") or []
    metrics: list[MetricItem] = []
    for m in raw_metrics[:3]:
        if isinstance(m, dict) and m.get("label") and m.get("value"):
            metrics.append(
                MetricItem(
                    label=_word_trim(str(m["label"]), 22),
                    value=_word_trim(str(m["value"]), 14),
                )
            )

    inf_type = str(raw.get("infographic_type") or "bars").lower().strip()
    if inf_type not in ("bars", "donut", "steps"):
        inf_type = "bars"

    inf_items: list[InfographicItem] = []
    for item in (raw.get("infographic_items") or [])[:4]:
        if isinstance(item, dict) and item.get("label"):
            try:
                pct = int(float(str(item.get("percent", 75)).replace("%", "")))
            except Exception:
                pct = 75
            pct = max(8, min(100, pct))
            inf_items.append(
                InfographicItem(
                    label=_word_trim(str(item["label"]), 26),
                    value=_word_trim(str(item.get("value") or f"{pct}%"), 14),
                    percent=pct,
                )
            )
    if not inf_items:
        inf_items = [
            InfographicItem(label=b.split(":")[0][:24] if ":" in b else _word_trim(b, 24), value=f"{88 - i*14}%", percent=88 - i * 14)
            for i, b in enumerate(bullets[:3])
        ]

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
        badge=_word_trim(str(raw.get("badge") or "FEATURED LAUNCH"), 34),
        headline=_word_trim(str(raw.get("headline") or "Built for Modern Teams"), 70),
        subhead=_word_trim(str(raw.get("subhead") or "Fast, reliable automation tailored to your workflow."), 135),
        bullets=bullets,
        icon_queries=icon_queries,
        metrics=metrics,
        infographic_type=inf_type,
        infographic_title=_word_trim(str(raw.get("infographic_title") or "Key Impact"), 32),
        infographic_items=inf_items,
        image_query=_word_trim(str(raw.get("image_query") or ""), 40) or None,
        cta=_word_trim(str(raw.get("cta") or "Get Started Today"), 34),
        footer_note=_word_trim(str(raw.get("footer_note") or ""), 45),
        accent_hex=accent,
    )


def _deterministic_condense(text: str, limit: int = 1800) -> str:
    return _word_trim(text, limit)


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
            model=Groq(id="openai/gpt-oss-20b", max_tokens=320),
            instructions=[
                "Condense the user's long product/event brief into 3-4 clean sentences (at most 140 words).",
                "Preserve: target audience, core value proposition, key numbers/stats, 3 benefits, and natural CTA/brand.",
                "Output plain prose sentences only—no markdown headers or 'Target Audience:' labels.",
            ],
        )
        res = summarizer.run(f"Condense this brief for a marketing poster:\n\n{capped}")
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
        svg_resp = await client.get(
            f"https://api.iconify.design/{prefix}/{name}.svg",
            params={"color": accent, "width": "24", "height": "24"},
        )
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


async def _search_wikimedia(client: httpx.AsyncClient, term: str) -> Optional[str]:
    resp = await client.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{term} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 5,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1080,
            "format": "json",
        },
    )
    pages = (resp.json().get("query") or {}).get("pages") or {}
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        thumb = info.get("thumburl") or info.get("url")
        clean_thumb = thumb.split("?")[0].lower() if thumb else ""
        if thumb and any(clean_thumb.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
            return thumb
    return None


async def _fetch_wikimedia_image(query: str) -> tuple[Optional[str], Optional[str], str]:
    try:
        async with httpx.AsyncClient(timeout=4.5, headers={"User-Agent": "PosterAIAgent/2.2 (https://chintu-poster-ai.vercel.app)"}) as client:
            for candidate in (query, query.split()[0] if query.split() else "technology", "abstract"):
                thumb = await _search_wikimedia(client, candidate)
                if thumb:
                    return thumb, "Wikimedia Commons (CC)", f"Matched copyright-free photo for '{candidate}'"
        return None, None, f"No Wikimedia bitmap match for '{query}'; using clean typographic layout"
    except Exception as exc:
        return None, None, f"Stock image lookup skipped ({type(exc).__name__})"


def _fallback_poster_spec(brief: str, preferred_template: Optional[str]) -> PosterSpec:
    cleaned = re.sub(r"\b(Target Audience|Core Value Proposition|Key Numbers/Stats|Benefits|CTA):\s*", "", brief, flags=re.I)
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", cleaned) if len(s.strip()) > 8]
    pct_match = re.search(r"(\d+%)", cleaned)
    domain_match = re.search(r"\b([a-z0-9-]+\.(?:ai|io|com|dev|app|org))\b", cleaned, re.I)
    metrics = [MetricItem(label="Key Impact", value=pct_match.group(1))] if pct_match else []
    inf_items = [
        InfographicItem(label="Efficiency Gain", value=pct_match.group(1) if pct_match else "80%", percent=85),
        InfographicItem(label="Automation Coverage", value="92%", percent=92),
        InfographicItem(label="Setup Speed", value="3x Faster", percent=78),
    ]
    return _clamp_spec(
        {
            "template": preferred_template or "dark-tech",
            "badge": "PRODUCT HIGHLIGHT",
            "headline": sentences[0] if sentences else "Automate Your Workflow Faster",
            "subhead": sentences[1] if len(sentences) > 1 else "Built for teams that want reliable, measurable results.",
            "bullets": sentences[2:5],
            "icon_queries": ["zap", "shield-check", "rocket"],
            "metrics": [m.model_dump() for m in metrics],
            "infographic_type": "bars",
            "infographic_title": "Performance Snapshot",
            "infographic_items": [item.model_dump() for item in inf_items],
            "image_query": "server",
            "cta": "Learn More",
            "footer_note": domain_match.group(1).lower() if domain_match else "",
            "accent_hex": "#3b82f6",
        },
        preferred_template,
    )


def _render_infographic_svg(
    spec: PosterSpec,
    fg: str,
    muted: str,
    card_bg: str,
    border: str,
) -> str:
    esc = html.escape
    accent = spec.accent_hex
    items = spec.infographic_items[:4]
    if not items:
        return ""

    if spec.infographic_type == "donut":
        primary = items[0]
        pct = max(5, min(100, primary.percent))
        r = 56
        circ = 2 * 3.14159 * r
        dash = circ * (pct / 100.0)
        gap = circ - dash
        legend_rows = "".join(
            f"""
            <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid {border};">
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="width:10px;height:10px;border-radius:999px;background:{accent if idx == 0 else muted};display:inline-block;"></span>
                <span style="font-size:19px;font-weight:600;color:{fg};">{esc(it.label)}</span>
              </div>
              <span style="font-size:19px;font-weight:800;color:{accent};">{esc(it.value)}</span>
            </div>
            """
            for idx, it in enumerate(items[:3])
        )
        return f"""
        <div style="padding:22px 26px;border-radius:20px;background:{card_bg};border:1px solid {border};">
          <div style="font-size:15px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{accent};margin-bottom:14px;">
            {esc(spec.infographic_title)}
          </div>
          <div style="display:flex;align-items:center;gap:28px;">
            <svg width="144" height="144" viewBox="0 0 144 144" style="flex-shrink:0;">
              <circle cx="72" cy="72" r="{r}" fill="none" stroke="{border}" stroke-width="14" />
              <circle cx="72" cy="72" r="{r}" fill="none" stroke="{accent}" stroke-width="14"
                stroke-linecap="round" stroke-dasharray="{dash:.1f} {gap:.1f}"
                transform="rotate(-90 72 72)" />
              <text x="72" y="79" text-anchor="middle" fill="{fg}" font-size="28" font-weight="800" font-family="Inter, sans-serif">{esc(primary.value)}</text>
            </svg>
            <div style="flex:1;">{legend_rows}</div>
          </div>
        </div>
        """

    if spec.infographic_type == "steps":
        step_cards = ""
        for idx, it in enumerate(items[:3]):
            arrow = f'<div style="font-size:24px;color:{accent};font-weight:800;padding:0 4px;">&rarr;</div>' if idx < min(len(items), 3) - 1 else ""
            step_cards += f"""
            <div style="flex:1;padding:16px;border-radius:16px;background:{accent}12;border:1px solid {accent}33;">
              <div style="font-size:14px;font-weight:800;color:{accent};margin-bottom:6px;">STEP {idx + 1}</div>
              <div style="font-size:20px;font-weight:700;color:{fg};line-height:1.25;">{esc(it.label)}</div>
              <div style="font-size:16px;font-weight:600;color:{muted};margin-top:6px;">{esc(it.value)}</div>
            </div>
            {arrow}
            """
        return f"""
        <div style="padding:22px 26px;border-radius:20px;background:{card_bg};border:1px solid {border};">
          <div style="font-size:15px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{accent};margin-bottom:14px;">
            {esc(spec.infographic_title)}
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            {step_cards}
          </div>
        </div>
        """

    # Default: "bars" horizontal comparison chart with crisp SVG bars
    bars_html = ""
    for it in items[:3]:
        bar_w = max(12, min(100, it.percent))
        bars_html += f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
            <span style="font-size:19px;font-weight:600;color:{fg};">{esc(it.label)}</span>
            <span style="font-size:19px;font-weight:800;color:{accent};">{esc(it.value)}</span>
          </div>
          <svg width="100%" height="14" viewBox="0 0 860 14" preserveAspectRatio="none" style="display:block;">
            <rect x="0" y="0" width="860" height="14" rx="7" fill="{border}" />
            <rect x="0" y="0" width="{int(8.6 * bar_w)}" height="14" rx="7" fill="{accent}" />
          </svg>
        </div>
        """
    return f"""
    <div style="padding:22px 26px;border-radius:20px;background:{card_bg};border:1px solid {border};">
      <div style="font-size:15px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{accent};margin-bottom:14px;">
        {esc(spec.infographic_title)}
      </div>
      {bars_html}
    </div>
    """


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
        <div style="width:200px;height:200px;border-radius:24px;background:{card_bg};border:1px solid {border};display:flex;align-items:center;justify-content:center;padding:14px;flex-shrink:0;">
          <img src="{hero_image_url}" alt="Product visual" style="max-width:100%;max-height:100%;object-fit:contain;" />
        </div>
        """

    infographic_html = _render_infographic_svg(spec, fg, muted, card_bg, border)

    bullet_html = "\n".join(
        f"""
        <div style="display:flex;align-items:center;gap:16px;padding:16px 22px;border-radius:18px;background:{card_bg};border:1px solid {border};">
          <div style="width:42px;height:42px;border-radius:12px;background:{accent}18;border:1px solid {accent}44;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            {icon_svgs[i] if i < len(icon_svgs) and icon_svgs[i] else f'<div style="width:12px;height:12px;border-radius:999px;background:{accent};"></div>'}
          </div>
          <div style="font-size:23px;line-height:1.32;font-weight:500;color:{fg};">{esc(b)}</div>
        </div>
        """
        for i, b in enumerate(spec.bullets)
    )

    credit_html = f'<span style="font-size:14px;color:{muted};display:block;margin-top:4px;">Photo: {esc(image_credit)}</span>' if image_credit else ""
    footer_right = (
        f'<div style="text-align:right;font-size:20px;color:{muted};font-weight:600;">{esc(spec.footer_note)}{credit_html}</div>'
        if (spec.footer_note or image_credit)
        else ""
    )

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
      padding: 68px;
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
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:24px;">
        <div style="flex:1;">
          <div style="display:inline-block;padding:9px 20px;border-radius:999px;background:{accent}22;border:1px solid {accent}66;color:{accent};font-size:18px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:22px;">
            {esc(spec.badge)}
          </div>
          <h1 style="font-size:60px;line-height:1.08;font-weight:800;letter-spacing:-0.03em;margin-bottom:18px;">
            {esc(spec.headline)}
          </h1>
          <p style="font-size:26px;line-height:1.38;color:{muted};max-width:860px;">
            {esc(spec.subhead)}
          </p>
        </div>
        {product_img_html}
      </div>
    </div>

    {infographic_html}

    <div style="display:flex;flex-direction:column;gap:12px;">
      {bullet_html}
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;gap:24px;padding-top:24px;border-top:1px solid {border};">
      <div style="padding:16px 30px;border-radius:16px;background:{accent};color:#ffffff;font-size:23px;font-weight:700;white-space:nowrap;">
        {esc(spec.cta)} &rarr;
      </div>
      {footer_right}
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
        "version": "2.2",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "summary_char_threshold": SUMMARY_CHAR_THRESHOLD,
    }


@app.post("/api/generate", response_model=GenerateResponse)
@app.post("/generate", response_model=GenerateResponse)
@app.post("/", response_model=GenerateResponse)
async def generate_poster(req: GenerateRequest):
    trace: list[TraceStep] = []
    original_chars = len(req.brief)

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
                model=Groq(
                    id="openai/gpt-oss-120b",
                    max_tokens=1500,
                ),
                instructions=[
                    "You design high-converting 1080x1350 infographic marketing posters. Output ONLY a valid JSON object.",
                    "Automatically choose the best visual theme ('dark-tech', 'clean-light', or 'bold-gradient') and accent_hex for the topic—no template preset is required.",
                    "Always include a rich visual infographic in the poster: pick infographic_type ('bars' for comparisons/metrics, 'donut' for percentage highlight, or 'steps' for a 3-step workflow), infographic_title (<=30 chars), and 3 infographic_items ([{label, value, percent (10-100)}]).",
                    "All JSON keys: template, badge (<=30 chars), headline (<=68 chars), subhead (<=130 chars), bullets (3 concise items <=75 chars each), icon_queries (3 Lucide icon names like 'zap', 'shield-check', 'rocket'), metrics (optional up to 3 {label, value}), infographic_type, infographic_title, infographic_items, image_query (1-2 word literal photo noun like 'laptop' or 'server'), cta (natural CTA from the user's brief <=30 chars), footer_note (brand/domain from the brief or empty string), accent_hex.",
                ],
            )
            prompt = f"Design a natural, infographic-rich poster for this brief:\n\n{effective_brief}"
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
                    detail=f"Generated '{spec.infographic_type}' infographic & PosterSpec via Agno + Groq (gpt-oss-120b)",
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

    t2 = time.perf_counter()
    poster_html = render_poster_html(spec, icon_svgs, hero_image_url, image_credit)
    trace.append(
        TraceStep(
            step="html_template_render",
            status="completed",
            duration_ms=int((time.perf_counter() - t2) * 1000),
            detail=f"Built sanitized 1080x1350 '{spec.template}' HTML with '{spec.infographic_type}' SVG infographic",
        )
    )

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
