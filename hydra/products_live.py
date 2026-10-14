"""Live product lookup from Open Beauty Facts (free, open, no key) plus user-added products.

Open Beauty Facts has ingredient lists, brand, images and barcodes. It does NOT have prices or stock,
so price is optional and entered by you. Coverage of Indian brands varies: if a product is missing,
add it yourself with the 'Add a product' form (copy the INCI ingredient list from the brand's site).
"""
import re
import uuid

from .ingredients import classify_many

OBF_SEARCH = "https://world.openbeautyfacts.org/cgi/search.pl"
OBF_PRODUCT = "https://world.openbeautyfacts.org/api/v2/product/{code}.json"
HEADERS = {"User-Agent": "Hydra.ai-student-project/1.0 (learning project)"}
FIELDS = ("code,product_name,brands,categories,ingredients_text,image_front_small_url,"
          "image_front_url,quantity,last_modified_t")

ROUTINE_CATEGORIES = ["cleanser", "serum", "treatment", "moisturizer", "sunscreen"]

_CATEGORY_WORDS = [
    ("sunscreen", r"sunscreen|sun cream|sun screen|\bspf\b|sun protection|sunblock"),
    ("cleanser", r"cleanser|face wash|facewash|micellar|cleansing|foaming wash"),
    ("serum", r"serum|ampoule|essence"),
    ("treatment", r"retinol|exfoliat|\bpeel\b|salicylic|glycolic|acne|spot treatment|\btoner\b"),
    ("moisturizer", r"moistur|face cream|gel cream|lotion|hydrating cream|\bcream\b"),
]
_CONCERNS_BY_ACTIVE = {
    "niacinamide": ["acne", "pigmentation", "redness"],
    "vitamin_c": ["pigmentation", "dullness", "aging"],
    "vitamin_c_derivative": ["pigmentation", "dullness"],
    "retinoid": ["aging", "acne", "pigmentation"],
    "azelaic": ["acne", "redness", "pigmentation"],
    "aha": ["dullness", "pigmentation", "aging"],
    "bha": ["acne"],
    "benzoyl_peroxide": ["acne"],
    "hyaluronic": ["dryness"],
    "ceramide": ["dryness", "redness"],
    "sunscreen": ["aging", "pigmentation"],
}
ALL_SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive"]


def split_ingredients(text: str) -> list[str]:
    """INCI text -> list of ingredient names."""
    if not text:
        return []
    text = re.sub(r"^\s*ingredients?\s*[:\-]\s*", "", text, flags=re.I)
    parts = re.split(r"[,;\n]+", text)
    out = []
    for p in parts:
        p = re.sub(r"[*_\[\]]", "", p).strip(" .\t")
        if 1 < len(p) <= 80:
            out.append(p)
    return out


def guess_category(name: str, categories: str, actives: set[str]) -> str | None:
    blob = f"{name} {categories}".lower()
    for cat, pat in _CATEGORY_WORDS:
        if re.search(pat, blob):
            return cat
    if actives & {"retinoid", "aha", "bha", "benzoyl_peroxide"}:
        return "treatment"
    if actives & {"vitamin_c", "vitamin_c_derivative", "niacinamide", "hyaluronic", "azelaic"}:
        return "serum"
    return None


def _build(pid, name, brand, category, ingredients, price, image_url, url, source, categories=""):
    actives = classify_many(ingredients)
    category = category or guess_category(name, categories, actives)
    concerns = sorted({c for a in actives for c in _CONCERNS_BY_ACTIVE.get(a, [])})
    if actives & {"retinoid", "aha", "bha", "benzoyl_peroxide"}:
        sessions = ["pm"]
    elif actives & {"vitamin_c", "sunscreen"}:
        sessions = ["am"]
    else:
        sessions = ["am", "pm"]
    return {
        "id": pid, "category": category, "name": name, "brand": brand,
        "price_inr": price, "sessions": sessions, "skin_types": list(ALL_SKIN_TYPES),
        "concerns": concerns, "ingredients": ingredients, "image_url": image_url,
        "url": url, "source": source, "auto_tagged": True, "demo": False,
    }


def normalise_obf(raw: dict) -> dict | None:
    """Open Beauty Facts product JSON -> Hydra product. Concerns and sessions are auto-tagged from ingredients."""
    code = str(raw.get("code") or "").strip()
    name = (raw.get("product_name") or "").strip()
    if not code or not name:
        return None
    image = raw.get("image_front_small_url") or raw.get("image_front_url") or None
    return _build(
        f"obf:{code}", name, (raw.get("brands") or "").split(",")[0].strip() or "Unknown brand", None,
        split_ingredients(raw.get("ingredients_text", "")), None,
        image if isinstance(image, str) and image.startswith("https://") else None,
        f"https://world.openbeautyfacts.org/product/{code}", "Open Beauty Facts",
        categories=raw.get("categories", ""),
    )


def from_user(name: str, brand: str, category: str, ingredients_text: str, price_inr, image_url=None, url=None) -> dict:
    name, brand = name.strip(), brand.strip() or "Unknown brand"
    if not name:
        raise ValueError("Product name is required")
    if category not in ROUTINE_CATEGORIES:
        raise ValueError(f"Category must be one of {', '.join(ROUTINE_CATEGORIES)}")
    ingredients = split_ingredients(ingredients_text)
    if not ingredients:
        raise ValueError("Paste the ingredient list (INCI), separated by commas")
    price = int(price_inr) if price_inr not in (None, "", 0) else None
    safe = lambda u: u if isinstance(u, str) and u.startswith("https://") else None
    return _build(f"custom:{uuid.uuid4().hex[:10]}", name, brand, category, ingredients, price,
                  safe(image_url), safe(url), "Added by you")


def _default_get(url, params=None):
    import requests
    r = requests.get(url, params=params, headers=HEADERS, timeout=12)
    r.raise_for_status()
    return r.json()


def search_live(query: str, limit: int = 12, http_get=None) -> list[dict]:
    http_get = http_get or _default_get
    data = http_get(OBF_SEARCH, {"search_terms": query, "search_simple": 1, "action": "process",
                                 "json": 1, "page_size": max(limit * 2, 20), "fields": FIELDS})
    out = []
    for raw in data.get("products", []):
        p = normalise_obf(raw)
        if p:
            out.append(p)
        if len(out) >= limit:
            break
    return out


def get_by_barcode(code: str, http_get=None) -> dict | None:
    http_get = http_get or _default_get
    data = http_get(OBF_PRODUCT.format(code=code), {"fields": FIELDS})
    raw = data.get("product")
    if not raw:
        return None
    raw = {**raw, "code": raw.get("code") or code}
    return normalise_obf(raw)
