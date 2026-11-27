# Poster AI Agent (Agno + Groq + Cloudflare Worker)

A lightweight, Vercel-ready AI poster generation pipeline built with **Agno**, **Groq**, **FastAPI**, and a **Cloudflare Worker** HTML-to-PNG renderer.

## Architecture

1. **Token-Window Summarization Guard (`_condense_with_agno`)**: If the incoming brief exceeds `2,000` characters, a lightweight `openai/gpt-oss-20b` Agno agent condenses it to `<= 150` words (with a deterministic fallback on rate limits) so the main agent never overflows Groq's context/TPM window.
2. **Structured Poster Agent (`openai/gpt-oss-120b`)**: Generates a validated Pydantic `PosterSpec` (`template`, `badge`, `headline`, `subhead`, `bullets`, `cta`) with JSON extraction & deterministic fallbacks.
3. **Deterministic `1080×1350` HTML & PNG Render**: Builds sanitized HTML and renders PNG via [`html-to-png-worker`](https://html-to-png-worker.chintusharma00014.workers.dev/).

## Deploy on Vercel

Set one environment variable in your Vercel project settings:
- `GROQ_API_KEY` — your Groq API key
