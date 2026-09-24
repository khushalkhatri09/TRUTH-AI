"""Tests for the TruthAI Flask app. The Gemini client is mocked — no real API calls."""

import os

# Set a dummy key BEFORE importing app. load_dotenv() does not override existing
# env vars, so this keeps tests independent of any real .env.
os.environ["GEMINI_API_KEY"] = "test-key"

import pytest  # noqa: E402

import app as truthai  # noqa: E402


VALID_ANALYSIS_JSON = """{
  "verdict": "FAKE",
  "credibility_score": 12,
  "confidence": 88,
  "summary": "s",
  "verdict_reasoning": "r",
  "red_flags": ["a"],
  "positive_signals": [],
  "writing_analysis": {"tone": "sensational", "clickbait_score": 9,
    "emotional_manipulation": 8, "factual_precision": 2, "grammar_quality": 5},
  "source_analysis": {"domain_trust": "low", "bias_detected": "none",
    "transparency_level": "low"},
  "claims_to_verify": ["c"],
  "recommended_sources": ["src"],
  "reader_tip": "tip"
}"""


class _FakeResult:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, text=None, exc=None):
        self._text, self._exc = text, exc

    def generate_content(self, **_kwargs):
        if self._exc:
            raise self._exc
        return _FakeResult(self._text)


class _FakeClient:
    def __init__(self, text=None, exc=None):
        self.models = _FakeModels(text, exc)


@pytest.fixture
def client():
    truthai.app.config["TESTING"] = True
    truthai.limiter.enabled = False  # don't let the 10/min cap interfere with tests
    with truthai.app.test_client() as c:
        yield c
    truthai.limiter.enabled = True


def _mock_gemini(monkeypatch, text=None, exc=None):
    monkeypatch.setattr(truthai, "get_client", lambda: _FakeClient(text=text, exc=exc))


# ── endpoints ─────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model"] == truthai.MODEL_NAME
    assert body["api_key_configured"] is True


def test_analyze_requires_min_length(client):
    r = client.post("/api/analyze", json={"text": "too short"})
    assert r.status_code == 400
    assert "40 characters" in r.get_json()["error"]


def test_analyze_success(client, monkeypatch):
    _mock_gemini(monkeypatch, text=VALID_ANALYSIS_JSON)
    r = client.post("/api/analyze", json={"text": "x" * 60})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["analysis"]["verdict"] == "FAKE"
    assert body["analysis"]["meta"]["model"] == truthai.MODEL_NAME


def test_analyze_extracts_json_from_prose(client, monkeypatch):
    _mock_gemini(monkeypatch, text="Sure! Here is the result:\n" + VALID_ANALYSIS_JSON + "\nThanks.")
    r = client.post("/api/analyze", json={"text": "x" * 60})
    assert r.status_code == 200
    assert r.get_json()["analysis"]["credibility_score"] == 12


def test_error_message_not_misclassified(client, monkeypatch):
    # A 404 whose text contains "generateContent" must NOT be labeled "Too many requests".
    _mock_gemini(monkeypatch, exc=RuntimeError(
        "404 models/x:generateContent is no longer available"))
    r = client.post("/api/analyze", json={"text": "x" * 60})
    assert r.status_code == 500
    err = r.get_json()["error"]
    assert "Too many requests" not in err
    assert "no longer available" in err


def test_analyze_url_path_uses_fetch(client, monkeypatch):
    monkeypatch.setattr(truthai, "fetch_article",
                        lambda url: {"title": "T", "body": "b" * 60, "source": "example.com"})
    _mock_gemini(monkeypatch, text=VALID_ANALYSIS_JSON)
    r = client.post("/api/analyze", json={"url": "http://example.com/a"})
    assert r.status_code == 200
    assert r.get_json()["analysis"]["meta"]["article_source"] == "example.com"


def test_transient_503_is_retried(client, monkeypatch):
    monkeypatch.setattr(truthai.time, "sleep", lambda *_a, **_k: None)  # no real waiting
    calls = {"n": 0}

    class _Flaky:
        def generate_content(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("503 UNAVAILABLE — high demand")
            return _FakeResult(VALID_ANALYSIS_JSON)

    class _FlakyClient:
        models = _Flaky()

    monkeypatch.setattr(truthai, "get_client", lambda: _FlakyClient())
    r = client.post("/api/analyze", json={"text": "x" * 60})
    assert r.status_code == 200
    assert calls["n"] == 3  # failed twice, succeeded on the third try


# ── SSRF guards ───────────────────────────────────────────────────────
@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "0.0.0.0"])
def test_private_hosts_blocked(host):
    with pytest.raises(ValueError):
        truthai._assert_public_host(host)


def test_public_host_allowed():
    # Should not raise for a public IP.
    truthai._assert_public_host("8.8.8.8")


@pytest.mark.parametrize("url", ["ftp://example.com", "file:///etc/passwd", "gopher://x"])
def test_safe_get_rejects_non_http(url):
    with pytest.raises(ValueError):
        truthai._safe_get(url)
