# Hydra.ai

AI skincare and AR makeup try-on. Scan your face, try on shades live through your webcam, search real products
(live from Open Beauty Facts), get a rule-checked routine, chat with a tool-calling agent on a FREE model, and earn
Hydra Points for sticking to your routine.

Everything runs locally in Python (FastAPI + OpenCV + MediaPipe). The browser only shows the UI.

## Setup (VS Code)

1. Install Python 3.10 to 3.12 (3.11 or 3.12 recommended for MediaPipe).
2. Open this folder in VS Code, then open a terminal and run:

```
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
3. Select the `.venv` interpreter in VS Code (Ctrl+Shift+P, "Python: Select Interpreter").
4. Copy `.env.example` to `.env`. For the chatbot pick ONE free option (or skip for offline mode):
   - **Gemini (free tier, cloud):** create a key at https://aistudio.google.com and set `GEMINI_API_KEY=` in `.env`.
   - **Ollama (free, fully local):** install https://ollama.com, run `ollama pull llama3.2`, keep Ollama running. No key needed.
   - **Groq (free tier, cloud):** key from https://console.groq.com, set `GROQ_API_KEY=`.
   - Anthropic Claude (paid) also works with `ANTHROPIC_API_KEY=`.
   Hydra picks the first one it finds. Force one with `HYDRA_PROVIDER=gemini` (or groq / ollama / anthropic / offline).
   Free-tier limits and model names change, so override with `HYDRA_MODEL=...` if a model is retired.
5. Run `python run.py`. The browser opens at http://127.0.0.1:8000.

First run downloads a ~4 MB face model into `models/` (needs internet once). If it is blocked, download
`face_landmarker.task` manually (the URL is printed in the error) and save it as `models/face_landmarker.task`.

## Debugging in VS Code

- Press F5 and pick "Hydra.ai (debug server)" (see `.vscode/launch.json`), then set breakpoints in `hydra/*.py`.
- "Hydra.ai tests" runs the test suite under the debugger.
- Health check: http://127.0.0.1:8000/api/health shows camera status, model status and whether Claude is configured.
- No webcam or tracking trouble? Use "Upload photo" in the Try-on tab, which runs the same analysis on an image.

## Project layout

```
run.py                  start the server
hydra/main.py           FastAPI routes + MJPEG video stream
hydra/camera.py         webcam thread: face tracking + AR overlays
hydra/tracker.py        MediaPipe Face Landmarker (Tasks API) + model download
hydra/tryon.py          lipstick and blush overlays (LAB colour blending)
hydra/skin.py           skin tone / undertone estimate, shade matching
hydra/ingredients.py    ingredient classes and conflict rules
hydra/routine.py        routine builder (analyst -> safety -> shopper -> coach)
hydra/agent.py          "Ask Hydra" tool-calling agent: Gemini / Groq / Ollama / Claude (+ offline fallback)
hydra/products_live.py  live Open Beauty Facts search + your own products, auto-tagged from ingredients
hydra/game.py           points, streaks, unlocks, badges (SQLite)
hydra/data/*.json       DEMO product catalogs (replace with real data)
hydra/static/index.html UI
tests/test_core.py      tests (no webcam or network needed)
```

## Troubleshooting

- "Could not open camera": close other apps using the webcam, allow camera permission, or set `HYDRA_CAMERA=1` in `.env`.
  Use the "Retry camera" button in the red banner afterwards.
- `ModuleNotFoundError`: the venv is not active or `pip install -r requirements.txt` did not finish.
- Overlay looks off or the skin tone seems wrong: scan in even, neutral light facing a window.
- Port busy: set `PORT=8001` in `.env`.
- Chat says the model is unavailable: check the key in `.env`, that Ollama is running (`ollama list`), or free-tier rate limits (HTTP 429). `/api/health` shows which provider was picked.
- Live search fails: check your internet connection; Open Beauty Facts can also be slow or briefly down.

## Real products

- Shop tab > "Search live" queries Open Beauty Facts (free, no key, needs internet) for real products and ingredient lists.
  Open the product, pick its type, optionally enter a price, and "Add to my routine pool". Added products are stored in
  `data/hydra.db` and are preferred over the demo products when building routines.
- Open Beauty Facts has **no prices or stock**, and Indian brand coverage varies. If a product is missing, use
  "Add a product you own" and paste the ingredient list from the brand's website.
- Concerns and AM/PM use for imported products are auto-tagged from their ingredients (a heuristic), and the UI says so.

## Honest limits

- Skin tone and undertone are heuristics from cheek pixels. Calibrate them on real faces in different lighting and report accuracy.
- The built-in catalog is demo data (tagged "demo"). Live prices or stock need a retailer feed, which this app does not have.
- Skincare output is cosmetic guidance, not medical advice.
- The video stream uses the webcam of the machine running the server (fine locally). For public deployment, capture in the browser and send frames to the API.
