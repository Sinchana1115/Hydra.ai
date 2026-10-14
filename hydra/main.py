"""Hydra.ai FastAPI app. Run with:  python run.py"""
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import catalog, products_live
from .agent import HydraAgent
from .camera import CameraService
from .config import DB_PATH, FACE_MODEL_PATH, STATIC_DIR
from .game import GameStore
from .ingredients import product_actives
from .routine import CONCERNS, SKIN_TYPES, build_routine, normalise_profile
from .skin import analyze_skin, match_foundations, recommend_by_fit
from .tracker import FaceTracker

store = GameStore(DB_PATH)
catalog.load_extra(store.list_products())  # real products you imported or added earlier
camera = CameraService()
agent = HydraAgent(store)
_static_tracker: Optional[FaceTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera.start()
    yield
    camera.stop()


app = FastAPI(title="Hydra.ai", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---- models --------------------------------------------------------
class TryOnIn(BaseModel):
    product_id: str


class ProfileIn(BaseModel):
    skin_type: str = "normal"
    concerns: List[str] = Field(default_factory=list)
    budget_inr: Optional[int] = None
    avoid_retinoids: bool = False


class ChatIn(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)


class ImportIn(BaseModel):
    code: str
    category: Optional[str] = None
    price_inr: Optional[int] = None


class CustomProductIn(BaseModel):
    name: str
    brand: str = ""
    category: str
    ingredients: str
    price_inr: Optional[int] = None
    image_url: Optional[str] = None
    url: Optional[str] = None


class CheckinIn(BaseModel):
    session: str


class UnlockIn(BaseModel):
    item_id: str


# ---- pages & stream --------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _mjpeg():
    while True:
        jpeg = camera.get_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.033)


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/health")
def health():
    return {
        "camera_running": camera.running,
        "camera_error": camera.error,
        "face_model_ready": FACE_MODEL_PATH.exists(),
        "chat_ready": agent.available,
        "provider": agent.provider,
        "model": agent.model,
    }


@app.post("/api/camera/restart")
def camera_restart():
    camera.restart()
    return {"ok": True}


# ---- try-on --------------------------------------------------------
@app.get("/api/tryon/catalog")
def tryon_catalog():
    unlocked = store.unlocked()
    items = []
    for p in catalog.makeup():
        if p["category"] in ("lipstick", "blush"):
            items.append({**p, "locked": p["unlock_cost"] > 0 and p["id"] not in unlocked})
    return {"items": items, "points": store.state()["points"]}


@app.post("/api/tryon")
def tryon(body: TryOnIn):
    p = catalog.makeup_by_id(body.product_id)
    if not p or p["category"] not in ("lipstick", "blush"):
        raise HTTPException(404, "Unknown try-on product")
    if p["unlock_cost"] > 0 and p["id"] not in store.unlocked():
        raise HTTPException(403, "Locked. Earn Hydra Points to unlock this shade.")
    if p["category"] == "lipstick":
        camera.set_style(lipstick=p["hex"])
    else:
        camera.set_style(blush=p["hex"])
    return {"ok": True}


@app.post("/api/tryon/clear")
def tryon_clear():
    camera.set_style(lipstick=None, blush=None)
    return {"ok": True}


# ---- scan ------------------------------------------------------------
def _scan_result(frame, landmarks):
    try:
        skin = analyze_skin(frame, landmarks)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    gained = store.record_scan(skin)
    return {
        "skin": skin,
        "foundations": match_foundations(skin, catalog.makeup_in("foundation")),
        "lipsticks": recommend_by_fit(skin, catalog.makeup_in("lipstick")),
        "blushes": recommend_by_fit(skin, catalog.makeup_in("blush")),
        "points_gained": gained,
    }


@app.post("/api/scan")
def scan():
    snap = camera.snapshot()
    if snap is None:
        raise HTTPException(422, "No face detected. Face the camera in even light and try again.")
    return _scan_result(*snap)


@app.post("/api/scan-photo")
async def scan_photo(image: UploadFile = File(...)):
    """Same analysis from an uploaded photo (handy for testing without a webcam)."""
    global _static_tracker
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "That file is not a readable image.")
    if _static_tracker is None:
        try:
            _static_tracker = FaceTracker(static=True)
        except Exception as exc:
            raise HTTPException(500, f"Face model unavailable: {exc}")
    landmarks = _static_tracker.process(frame)
    if landmarks is None:
        raise HTTPException(422, "No face detected in that photo.")
    return _scan_result(frame, landmarks)


# ---- profile & routine -----------------------------------------------
@app.get("/api/profile")
def get_profile():
    return {"profile": normalise_profile(store.get_profile()), "latest_scan": store.latest_scan(),
            "skin_types": SKIN_TYPES, "concerns": CONCERNS}


@app.post("/api/profile")
def set_profile(body: ProfileIn):
    profile = normalise_profile(body.model_dump())
    store.set_profile(profile)
    return {"profile": profile}


@app.get("/api/routine")
def routine():
    return build_routine(store.get_profile())


# ---- products ----------------------------------------------------------
def _decorate(products):
    have = {p["id"] for p in catalog.skincare()}
    return [{**p, "in_catalog": p["id"] in have, "actives": sorted(product_actives(p))} for p in products]


@app.get("/api/products/local")
def products_local():
    return {"items": _decorate(catalog.skincare())}


@app.get("/api/products/search")
def products_search(q: str, live: bool = True):
    q = q.strip()
    if len(q) < 2:
        raise HTTPException(400, "Type at least 2 characters.")
    if not live:
        ql = q.lower()
        return {"source": "local", "items": _decorate(
            [p for p in catalog.skincare() if ql in f"{p['name']} {p['brand']} {p['category']}".lower()])}
    try:
        return {"source": "Open Beauty Facts (live)", "items": _decorate(products_live.search_live(q))}
    except Exception as exc:
        raise HTTPException(502, f"Live search failed ({type(exc).__name__}). Check your internet connection and try again.")


@app.post("/api/products/import")
def products_import(body: ImportIn):
    """Fetch the product fresh from Open Beauty Facts by barcode and add it to your catalog."""
    try:
        p = products_live.get_by_barcode(body.code.replace("obf:", ""))
    except Exception as exc:
        raise HTTPException(502, f"Live lookup failed ({type(exc).__name__}). Try again.")
    if p is None:
        raise HTTPException(404, "Product not found.")
    if body.category:
        if body.category not in products_live.ROUTINE_CATEGORIES:
            raise HTTPException(400, "Unknown category")
        p["category"] = body.category
    if p["category"] is None:
        raise HTTPException(422, "Could not tell what type of product this is. Pick a category and import again.")
    if not p["ingredients"]:
        raise HTTPException(422, "This product has no ingredient list in the database, so it cannot be used.")
    if body.price_inr:
        p["price_inr"] = body.price_inr
    store.save_product(p)
    catalog.add_extra(p)
    return {"ok": True, "product": p}


@app.post("/api/products/custom")
def products_custom(body: CustomProductIn):
    try:
        p = products_live.from_user(body.name, body.brand, body.category, body.ingredients,
                                    body.price_inr, body.image_url, body.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    store.save_product(p)
    catalog.add_extra(p)
    return {"ok": True, "product": p}


@app.delete("/api/products/{product_id:path}")
def products_delete(product_id: str):
    if not catalog.remove_extra(product_id):
        raise HTTPException(404, "Only products you imported or added can be removed.")
    store.delete_product(product_id)
    return {"ok": True}


# ---- chat ------------------------------------------------------------
@app.post("/api/chat")
def chat(body: ChatIn):
    return agent.chat(body.message, body.history)


# ---- game ------------------------------------------------------------
@app.get("/api/game")
def game():
    return store.state()


@app.post("/api/checkin")
def checkin(body: CheckinIn):
    try:
        return store.checkin(body.session)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/unlock")
def unlock(body: UnlockIn):
    p = catalog.makeup_by_id(body.item_id)
    if not p:
        raise HTTPException(404, "Unknown item")
    return store.unlock(p["id"], p["unlock_cost"])
