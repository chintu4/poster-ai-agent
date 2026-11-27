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

app = FastAPI(title="Poster AI Agent (Agno + Groq)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    brief: str = Field(..., min_length=5, max_length=20000)
    template: Optional[Literal["dark-tech", "clean-light", "bold-gradient"]] = None
    render_png: bool = True


class PosterSpec(BaseModel):
    template: Literal["dark-tech", "clean-light", "bold-gradient"] = "dark-tech"
    badge: str = Field(..., max_length=40)
    headline: str = Field(..., max_length=80)
    subhead: str = Field(..., max_length=160)
    bullets: list[str] = Field(..., min_length=2, max_length=4)
    cta: str = Field(..., max_length=45)
    footer_note: str = Field(default="chintu-work.vercel.app", max_length=60)
    accent_hex: str = Field(default="#3b82f6")


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
            "GROQ_API_KEY missing; used deterministic head condensation",
        )

    try:
        from agno.agent import Agent
        from agno.models.groq import Groq

        summarizer = Agent(
            name="Brief Summarizer",
            model=Groq(id="openai/gpt-oss-20b", max_tokens=280),
            instructions=[
                "Condense the user's long product/event brief into at most 150 words.",
                "Preserve: target audience, core value proposition, 3 key benefits, and call to action.",
                "Output plain text only—no markdown headers or filler.",
            ],
        )
        res = summarizer.run(
            f"Condense this brief for a 1080x1350 marketing poster:\n\n{capped}"
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


def _fallback_poster_spec(brief: str, preferred_template: Optional[str]) -> PosterSpec:
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", brief) if len(s.strip()) > 6]
    headline = sentences[0][:68] if sentences else "Launch Faster with Autonomous AI Agents"
    subhead = (
        sentences[1][:145]
        if len(sentences) > 1
        else "Production-ready workflows built with Agno, Groq, and clean APIs."
    )
    bullets = [s[:72] for s in sentences[2:5]]
    while len(bullets) < 3:
        defaults = [
            "Sub-second inference & deterministic validation",
            "Server-side key safety & graceful rate-limit guards",
            "Ready to deploy & integrate into your stack",
        ]
        bullets.append(defaults[len(bullets)])

    return PosterSpec(
        template=preferred_template or "dark-tech",
        badge="AI AGENT SHOWCASE",
        headline=headline,
        subhead=subhead,
        bullets=bullets[:3],
        cta="Book a 15-min Architecture Call",
        footer_note="chintu-work.vercel.app",
        accent_hex="#3b82f6",
    )


def render_poster_html(spec: PosterSpec) -> str:
    esc = html.escape
    accent = spec.accent_hex if re.match(r"^#[0-9a-fA-F]{6}$", spec.accent_hex) else "#3b82f6"

    if spec.template == "clean-light":
        bg = "#ffffff"
        fg = "#09090b"
        muted = "#52525b"
        card_bg = "#f4f4f5"
        border = "#e4e4e7"
    elif spec.template == "bold-gradient":
        bg = f"linear-gradient(145deg, #0f172a 0%, #1e1b4b 55%, {accent}33 100%)"
        fg = "#f8fafc"
        muted = "#cbd5e1"
        card_bg = "rgba(255,255,255,0.06)"
        border = "rgba(255,255,255,0.14)"
    else:
        bg = "#09090b"
        fg = "#fafafa"
        muted = "#a1a1aa"
        card_bg = "#18181b"
        border = "#27272a"

    bullet_html = "\n".join(
        f"""
        <div style="display:flex;align-items:flex-start;gap:18px;padding:22px 26px;border-radius:20px;background:{card_bg};border:1px solid {border};">
          <div style="width:14px;height:14px;border-radius:999px;background:{accent};margin-top:10px;flex-shrink:0;"></div>
          <div style="font-size:28px;line-height:1.38;font-weight:500;color:{fg};">{esc(b)}</div>
        </div>
        """
        for b in spec.bullets
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: 1080px;
      height: 1350px;
      background: {bg};
      color: {fg};
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      padding: 84px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <div>
    <div style="display:inline-block;padding:12px 24px;border-radius:999px;background:{accent}22;border:1px solid {accent}66;color:{accent};font-size:22px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:36px;">
      {esc(spec.badge)}
    </div>
    <h1 style="font-size:74px;line-height:1.08;font-weight:800;letter-spacing:-0.03em;margin-bottom:28px;">
      {esc(spec.headline)}
    </h1>
    <p style="font-size:32px;line-height:1.45;color:{muted};max-width:880px;">
      {esc(spec.subhead)}
    </p>
  </div>

  <div style="display:flex;flex-direction:column;gap:20px;">
    {bullet_html}
  </div>

  <div style="display:flex;align-items:center;justify-content:space-between;padding-top:36px;border-top:1px solid {border};">
    <div style="padding:22px 38px;border-radius:18px;background:{accent};color:#ffffff;font-size:28px;font-weight:700;">
      {esc(spec.cta)} &rarr;
    </div>
    <div style="font-size:24px;color:{muted};font-weight:600;">
      {esc(spec.footer_note)}
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
def health():
    return {
        "status": "ok",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "summary_char_threshold": SUMMARY_CHAR_THRESHOLD,
    }


@app.post("/api/generate", response_model=GenerateResponse)
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

    t1 = time.perf_counter()
    mode: Literal["live", "fallback"] = "live"
    spec: Optional[PosterSpec] = None
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if api_key:
        try:
            from agno.agent import Agent
            from agno.models.groq import Groq

            poster_agent = Agent(
                name="Poster Copy & Layout Agent",
                model=Groq(id="openai/gpt-oss-120b", max_tokens=650),
                output_schema=PosterSpec,
                instructions=[
                    "Convert the user's brief into a crisp 1080x1350 marketing poster specification.",
                    "Return valid JSON matching PosterSpec: template ('dark-tech', 'clean-light', or 'bold-gradient'), badge, headline, subhead, bullets (3 concise items), cta, footer_note, accent_hex.",
                    "Keep copy punchy, concrete, and free of fluff.",
                ],
            )
            result = poster_agent.run(
                f"Preferred template: {req.template or 'auto'}\n\nBrief:\n{effective_brief}"
            )
            content = getattr(result, "content", None)
            if isinstance(content, PosterSpec):
                spec = content
            elif isinstance(content, dict):
                spec = PosterSpec.model_validate(content)
            elif isinstance(content, str):
                spec = PosterSpec.model_validate(_extract_json_block(content))

            if spec and req.template:
                spec.template = req.template

            trace.append(
                TraceStep(
                    step="agno_poster_agent",
                    status="completed",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    detail="Generated & validated PosterSpec via Agno + Groq (gpt-oss-120b)",
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

    t2 = time.perf_counter()
    poster_html = render_poster_html(spec)
    trace.append(
        TraceStep(
            step="html_template_render",
            status="completed",
            duration_ms=int((time.perf_counter() - t2) * 1000),
            detail=f"Built sanitized 1080x1350 '{spec.template}' HTML",
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
