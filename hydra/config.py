import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load ANTHROPIC_API_KEY etc. from .env
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "hydra.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# Chat providers. Gemini, Groq and Ollama all speak the OpenAI-compatible chat API.
PROVIDERS = {
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
    "groq": {"base": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
    "ollama": {"base": os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1",
               "key_env": None, "model": "llama3.2"},
    "anthropic": {"base": None, "key_env": "ANTHROPIC_API_KEY", "model": "claude-sonnet-5-5"},
}
MODEL_OVERRIDE = os.getenv("HYDRA_MODEL") or None
PROVIDER_OVERRIDE = (os.getenv("HYDRA_PROVIDER") or "").strip().lower() or None
CAMERA_INDEX = int(os.getenv("HYDRA_CAMERA", "0"))
