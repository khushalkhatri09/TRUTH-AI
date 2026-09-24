"""
TruthAI — AI-Powered Fake News Detector (Google Gemini)
Python / Flask port of the original Node.js (Express) server.
"""

import os
import re
import time
import json
import socket
import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Load environment (.env) ───────────────────────────────────────────
load_dotenv()

PORT = int(os.environ.get("PORT", 3000))
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
MODEL_NAME = "gemini-flash-latest"  # stable alias — always the current Flash model

# ── Security / fetch limits ───────────────────────────────────────────
# Comma-separated list of extra cross-origin sites allowed to call the API.
# Empty (default) = same-origin only, which is all the bundled frontend needs.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10 per minute")
FETCH_TIMEOUT = 12                     # seconds
MAX_FETCH_BYTES = 2 * 1024 * 1024      # 2 MB cap on fetched article HTML
MAX_REDIRECTS = 3
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# A full browser-like header set. Many news sites (NDTV, etc.) return 403 to
# requests that send only a User-Agent, so we mimic a real Chrome request.
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# ── Flask app (serves ./public as static, like express.static) ────────
app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")

# CORS: only enable cross-origin access when explicitly configured.
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# Rate limiting: per-client-IP cap on the expensive endpoint.
limiter = Limiter(key_func=get_remote_address, app=app)

# ── Provider config (Gemini and/or OpenAI, each with key rotation) ────
# The app can talk to Google Gemini OR OpenAI. Choose with PROVIDER:
#   PROVIDER=openai | gemini | auto      (default: auto)
# "auto" picks OpenAI if an OpenAI key is set, otherwise Gemini.
# Each provider supports a POOL of keys — when one hits its rate limit (429)
# or is rejected, the app rotates to the next key automatically. Provide via
# (merged, in order, de-duplicated):
#   GEMINI_API_KEYS=k1,k2   (+ GEMINI_API_KEY  for a single key)
#   OPENAI_API_KEYS=k1,k2   (+ OPENAI_API_KEY  for a single key)
def _load_keys(*env_names):
    raw = []
    for name in env_names:
        raw += os.environ.get(name, "").split(",")
    keys, seen = [], set()
    for k in raw:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


GEMINI_KEYS = _load_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
OPENAI_KEYS = _load_keys("OPENAI_API_KEYS", "OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# OPENAI_MODELS: optional comma-separated fallback list. If the first model
# is busy/rate-limited, the app tries the next one (great for free pools like
# OpenRouter where any single model is often temporarily overloaded).
OPENAI_MODELS = [m.strip() for m in
                 (os.environ.get("OPENAI_MODELS") or OPENAI_MODEL).split(",") if m.strip()]
# Optional: point the OpenAI-compatible client at another provider (Groq,
# OpenRouter, Cerebras, Mistral, …). Leave empty to use OpenAI itself.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()


def _resolve_provider():
    p = os.environ.get("PROVIDER", "auto").strip().lower()
    if p in ("openai", "gemini"):
        return p
    return "openai" if OPENAI_KEYS else "gemini"


PROVIDER = _resolve_provider()
ACTIVE_KEYS = OPENAI_KEYS if PROVIDER == "openai" else GEMINI_KEYS
ACTIVE_MODELS = OPENAI_MODELS if PROVIDER == "openai" else [MODEL_NAME]
ACTIVE_MODEL = ACTIVE_MODELS[0]  # primary, for display/metadata
if PROVIDER == "openai":
    PROVIDER_LABEL = f"OpenAI-compatible ({urlparse(OPENAI_BASE_URL).hostname})" if OPENAI_BASE_URL else "OpenAI"
else:
    PROVIDER_LABEL = "Google Gemini"

# Backward-compat alias (health/banner still reference API_KEYS).
API_KEYS = ACTIVE_KEYS

SYSTEM_INSTRUCTION = (
    "You are TruthAI, an elite fact-checking system. Always respond "
    "ONLY with valid raw JSON — no markdown, no code fences, no extra text."
)

_clients = {}       # api_key -> provider client (built lazily, then cached)
_key_index = 0      # sticky pointer to the key that last succeeded


class _Result:
    """Uniform wrapper so downstream code can always read `.text`."""
    def __init__(self, text):
        self.text = text


def _client_for(api_key):
    client = _clients.get(api_key)
    if client is None:
        if PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL or None)
        else:
            client = genai.Client(api_key=api_key)
        _clients[api_key] = client
    return client


def _is_quota_error(err):
    raw = str(err)
    low = raw.lower()
    return ("429" in raw or "resource_exhausted" in low or "quota" in low
            or "rate limit" in low or "rate_limit" in low)


def _is_auth_error(err):
    raw = str(err)
    low = raw.lower()
    return ("api_key_invalid" in low or "api key not valid" in low
            or "permission_denied" in low or "invalid_api_key" in low
            or "incorrect api key" in low or "401" in raw)


def _is_transient(err):
    raw = str(err)
    low = raw.lower()
    return ("503" in raw or "500" in raw or "502" in raw
            or "unavailable" in low or "overloaded" in low
            or "temporarily rate-limited" in low  # OpenRouter shared free pool
            or "provider returned error" in low)


# Providers occasionally return transient 5xx errors. Retry a couple of
# times with backoff so users don't see a spurious failure.
TRANSIENT_RETRIES = 2


def _call_model(client, prompt, model):
    """One provider call for a specific model → uniform _Result with .text.

    We deliberately do NOT send response_format={"type":"json_object"} —
    many free/open models (and their upstream providers) reject it. The
    system prompt already demands raw JSON, and the caller extracts the
    JSON object with a regex, so this stays compatible everywhere.
    """
    if PROVIDER == "openai":
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return _Result(resp.choices[0].message.content)
    # Gemini
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=2048,
        temperature=0.2,
    )
    resp = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
    return _Result(resp.text)


def generate_analysis(prompt):
    """Call the active provider, rotating through its key pool on failures.

    Start from the last key that worked (sticky). For each key, retry
    transient 5xx errors with backoff. On a quota (429) or auth error,
    rotate immediately to the next key. Only when every key has failed with
    a quota/auth error do we surface the error to the caller.
    """
    global _key_index
    if not ACTIVE_KEYS:
        raise ValueError(f"No API key set for provider '{PROVIDER}'.")

    n = len(ACTIVE_KEYS)
    last_err = None
    for offset in range(n):
        idx = (_key_index + offset) % n
        client = _client_for(ACTIVE_KEYS[idx])
        auth_failed = False
        # Try each candidate model in turn; within a model, retry transient
        # / busy errors with backoff before falling back to the next model.
        for model in ACTIVE_MODELS:
            for attempt in range(TRANSIENT_RETRIES + 1):
                try:
                    result = _call_model(client, prompt, model)
                    _key_index = idx  # remember the working key for next time
                    return result
                except Exception as err:  # noqa: BLE001
                    # Busy / transient / shared-pool 429 → wait and retry.
                    if (_is_transient(err) or _is_quota_error(err)) and attempt < TRANSIENT_RETRIES:
                        time.sleep(1.5 * (attempt + 1))  # 1.5s, then 3s
                        continue
                    if _is_auth_error(err):
                        last_err = err
                        auth_failed = True
                        break  # bad key — stop trying models, rotate key
                    if _is_quota_error(err) or _is_transient(err):
                        last_err = err
                        if len(ACTIVE_MODELS) > 1:
                            print(f"[TruthAI] '{model}' busy/rate-limited — trying next model.")
                        break  # fall back to the next model
                    raise  # a real error (bad request, network, etc.) — surface it
            if auth_failed:
                break  # rotate to the next key
    # Nothing succeeded across all keys × models.
    raise last_err or RuntimeError(
        "All models are rate-limited right now. Please try again in a moment."
    )


# ── SSRF protection ───────────────────────────────────────────────────
def _assert_public_host(hostname):
    """Reject hosts that resolve to private/loopback/link-local/reserved IPs."""
    if not hostname:
        raise ValueError("Invalid URL: missing host.")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError("Could not resolve the URL host.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError("Refusing to fetch a non-public (internal) address.")


def _safe_get(url):
    """GET a URL with SSRF guards: scheme allowlist, private-IP block on every
    hop, manual redirect following, and a hard response-size cap."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http(s) URLs are allowed.")
        _assert_public_host(parsed.hostname)

        resp = requests.get(
            current,
            timeout=FETCH_TIMEOUT,
            allow_redirects=False,          # follow manually so each hop is validated
            stream=True,                    # so we can enforce the size cap
            headers=BROWSER_HEADERS,
        )

        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise ValueError("Received a redirect with no location.")
            current = urljoin(current, location)
            continue

        resp.raise_for_status()

        total = 0
        chunks = []
        for chunk in resp.iter_content(8192):
            total += len(chunk)
            if total > MAX_FETCH_BYTES:
                resp.close()
                raise ValueError("Article is too large to fetch.")
            chunks.append(chunk)
        encoding = resp.encoding or "utf-8"
        resp.close()
        return b"".join(chunks).decode(encoding, errors="replace")

    raise ValueError("Too many redirects.")


# ── Fetch article from URL ────────────────────────────────────────────
def fetch_article(url):
    html = _safe_get(url)
    soup = BeautifulSoup(html, "html.parser")

    # Strip noise (equivalent to cheerio's .remove())
    for el in soup.select("script,style,nav,footer,header,aside,.ad,.advertisement,.sidebar"):
        el.decompose()

    # Title: first <h1>, else og:title, else <title>
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Paragraphs (keep only substantial ones)
    paragraphs = []
    for el in soup.select("article p, main p, .post-content p, .entry-content p, p"):
        t = el.get_text(strip=True)
        if len(t) > 50:
            paragraphs.append(t)

    body = "\n\n".join(paragraphs[:40])
    if not body:
        raise ValueError("Could not extract article text from this URL.")

    return {"title": title, "body": body, "source": urlparse(url).hostname}


# ── Analysis prompt ───────────────────────────────────────────────────
def build_prompt(title, body, source):
    return f"""You are TruthAI, an elite fact-checking and misinformation detection system.
Analyze the following news article with maximum precision.

TITLE: {title or "Not provided"}
SOURCE DOMAIN: {source or "Unknown"}
CONTENT:
\"\"\"
{body[:5000]}
\"\"\"

Return ONLY a single valid JSON object. No markdown. No code fences. No extra text. Exactly this structure:

{{
  "verdict": "REAL" | "FAKE" | "MISLEADING" | "SATIRE" | "UNVERIFIED",
  "credibility_score": <0-100 integer>,
  "confidence": <0-100 integer>,
  "summary": "<3 sentence plain-English analysis of the article>",
  "verdict_reasoning": "<Why you assigned this verdict>",
  "red_flags": ["<flag 1>", "<flag 2>", "<flag 3>"],
  "positive_signals": ["<signal 1>", "<signal 2>"],
  "writing_analysis": {{
    "tone": "<neutral|sensational|emotional|balanced|manipulative>",
    "clickbait_score": <0-10>,
    "emotional_manipulation": <0-10>,
    "factual_precision": <0-10>,
    "grammar_quality": <0-10>
  }},
  "source_analysis": {{
    "domain_trust": "<high|medium|low|unknown>",
    "bias_detected": "<left|right|center|none|unknown>",
    "transparency_level": "<high|medium|low>"
  }},
  "claims_to_verify": ["<key claim 1>", "<key claim 2>", "<key claim 3>"],
  "recommended_sources": ["<trusted source to cross check>", "<another source>"],
  "reader_tip": "<one actionable verification tip for this specific article>"
}}"""


# ── POST /api/analyze ─────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    url = data.get("url")

    if not API_KEYS:
        return jsonify({
            "error": "No API key set. Add GEMINI_API_KEY (or GEMINI_API_KEYS) "
                     "to your .env file. Get one at: https://aistudio.google.com/apikey",
        }), 400

    try:
        article_title = ""
        article_body = text or ""
        article_source = "Pasted Text"

        if url and url.startswith("http"):
            fetched = fetch_article(url)
            article_title = fetched["title"]
            article_body = fetched["body"]
            article_source = fetched["source"]

        if not article_body or len(article_body.strip()) < 40:
            return jsonify({
                "error": "Please provide at least 40 characters of article text."
            }), 400

        result = generate_analysis(
            build_prompt(article_title, article_body, article_source)
        )

        raw_text = result.text

        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            raise ValueError("AI returned an unexpected response. Please try again.")

        analysis = json.loads(match.group(0))
        analysis["meta"] = {
            "article_title": article_title,
            "article_source": article_source,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "provider": PROVIDER_LABEL,
            "model": ACTIVE_MODEL,
        }

        return jsonify({"success": True, "analysis": analysis})

    except Exception as err:  # noqa: BLE001 — mirror the JS catch-all
        raw = str(err) or "Analysis failed."
        print(f"[TruthAI Error] {raw}")

        low = raw.lower()
        # Precise checks — avoid matching substrings like "rate" inside "generateContent".
        if "403" in raw or "forbidden" in low or "401" in raw and "url" in low:
            msg = ("This site blocked automated access (403). It has bot "
                   "protection — please copy the article text and paste it "
                   "into the text box instead of using the URL.")
        elif _is_auth_error(err):
            msg = f"Invalid API key. Check your {PROVIDER.upper()}_API_KEY in .env."
        elif "429" in raw or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
            msg = "Too many requests. Please wait a moment and try again."
        elif "503" in raw or "unavailable" in low or "overloaded" in low:
            msg = "AI service is temporarily busy. Please try again in a few seconds."
        else:
            # Surface the real error instead of hiding it behind a generic message.
            msg = raw

        return jsonify({"error": msg}), 500


# ── 429 handler (keep the JSON error shape the frontend expects) ──────
@app.errorhandler(429)
def ratelimit_handler(_e):
    return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429


# ── GET /api/health ───────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "app": "TruthAI v1.0.0",
        "provider": PROVIDER_LABEL,
        "model": ACTIVE_MODEL,
        "api_key_configured": bool(ACTIVE_KEYS),
        "api_keys_count": len(ACTIVE_KEYS),
    })


# ── Serve the frontend (index.html) ───────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


# ── Start ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner = r"""
  ████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗ █████╗ ██╗
  ╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║██╔══██╗██║
     ██║   ██████╔╝██║   ██║   ██║   ███████║███████║██║
     ██║   ██╔══██╗██║   ██║   ██║   ██╔══██║██╔══██║██║
     ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║██║  ██║██║
     ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
"""
    print(banner)
    print(f"  TruthAI ({PROVIDER_LABEL}) → http://localhost:{PORT}")
    print(f"  Model: {ACTIVE_MODEL}")
    if ACTIVE_KEYS:
        key_status = f"{len(ACTIVE_KEYS)} {PROVIDER} key(s) loaded — auto-rotation on rate limit"
    else:
        key_status = f"Missing — add {PROVIDER.upper()}_API_KEY (or {PROVIDER.upper()}_API_KEYS) to .env"
    print(f"  API Keys: {key_status}\n")

    # threaded=True so concurrent requests behave like Node's async server
    app.run(host="0.0.0.0", port=PORT, threaded=True)
