import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"
_EXTRA: list[dict] = []  # imported / user-added real products (persisted by GameStore)


@lru_cache(maxsize=None)
def makeup() -> list[dict]:
    return json.loads((_DATA / "makeup_catalog.json").read_text(encoding="utf-8"))["items"]


@lru_cache(maxsize=None)
def _skincare_base() -> list[dict]:
    items = json.loads((_DATA / "skincare_catalog.json").read_text(encoding="utf-8"))["items"]
    return [{**p, "demo": True} for p in items]


def skincare() -> list[dict]:
    return _skincare_base() + _EXTRA


def load_extra(products: list[dict]) -> None:
    _EXTRA.clear()
    _EXTRA.extend(products)


def add_extra(product: dict) -> None:
    _EXTRA[:] = [p for p in _EXTRA if p["id"] != product["id"]] + [product]


def remove_extra(product_id: str) -> bool:
    before = len(_EXTRA)
    _EXTRA[:] = [p for p in _EXTRA if p["id"] != product_id]
    return len(_EXTRA) < before


def skincare_by_id(product_id: str) -> dict | None:
    return next((p for p in skincare() if p["id"] == product_id), None)


def makeup_by_id(item_id: str) -> dict | None:
    return next((x for x in makeup() if x["id"] == item_id), None)


def makeup_in(category: str) -> list[dict]:
    return [x for x in makeup() if x["category"] == category]
