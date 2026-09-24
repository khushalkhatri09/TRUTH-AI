# 🛡️ TruthAI — AI Fake News Detector (Google Gemini)

> Powered by Gemini 2.0 Flash · Credibility scoring · Source analysis

---

## 🚀 Quick Start (3 Steps)

### Step 1 — Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Add your Gemini API key
```bash
cp .env.example .env
```
Open `.env` and set:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your key FREE at: https://aistudio.google.com/apikey

### Step 3 — Run the app
```bash
python app.py
```
Open browser → **http://localhost:3000**

---

## 🔧 Commands

| Command              | Description                          |
|----------------------|--------------------------------------|
| `python app.py`      | Start the app                        |
| `flask --app app run --debug` | Start with auto-reload (dev mode) |

---

## ✅ Features
- 📝 Paste text or 🔗 enter a URL
- 📊 Credibility score (0–100) with animated gauge
- 🚩 Red flags & ✅ positive signals
- ✍️ Writing analysis: clickbait, manipulation, grammar
- 🌐 Source trust, bias, transparency breakdown
- 💡 Reader verification tips

---

## 📁 Project Structure
```
TruthAI/
├── app.py            ← Flask + Gemini backend
├── public/
│   └── index.html    ← Frontend UI
├── .env              ← Your API key (create from .env.example)
├── .env.example      ← Template
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🔌 API

| Method | Route          | Description                                  |
|--------|----------------|----------------------------------------------|
| `POST` | `/api/analyze` | Body: `{ "text": "..." }` or `{ "url": "..." }` |
| `GET`  | `/api/health`  | Service status + whether the API key is set  |
