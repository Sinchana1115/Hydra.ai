"""Routine builder: analyst -> safety -> shopper -> coach, as plain functions (they map 1:1 to LangGraph nodes later)."""
from . import catalog
from .ingredients import check_products, notes_for, product_actives

CONCERNS = ["acne", "pigmentation", "aging", "dullness", "redness", "dryness"]
SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive"]
SENSITIVE_BLOCK = {"retinoid", "aha", "bha", "vitamin_c", "benzoyl_peroxide"}
DISCLAIMER = "Cosmetic guidance only, not medical advice. For persistent or severe skin issues, see a dermatologist."

DEFAULT_PROFILE = {"skin_type": "normal", "concerns": [], "budget_inr": None, "avoid_retinoids": False}


def normalise_profile(profile: dict | None) -> dict:
    p = {**DEFAULT_PROFILE, **(profile or {})}
    if p["skin_type"] not in SKIN_TYPES:
        p["skin_type"] = "normal"
    p["concerns"] = [c for c in p.get("concerns", []) if c in CONCERNS]
    budget = p.get("budget_inr")
    p["budget_inr"] = int(budget) if budget else None
    p["avoid_retinoids"] = bool(p.get("avoid_retinoids"))
    return p


def _eligible(product: dict, profile: dict, session: str) -> bool:
    if session not in product["sessions"]:
        return False
    if profile["skin_type"] not in product["skin_types"]:
        return False
    price = product.get("price_inr")
    if profile["budget_inr"] and price is not None and price > profile["budget_inr"]:
        return False
    actives = product_actives(product)
    if profile["skin_type"] == "sensitive" and actives & SENSITIVE_BLOCK:
        return False
    if profile["avoid_retinoids"] and "retinoid" in actives:
        return False
    return True


def _pick(products, category, session, profile, concerns, need_match=False):
    pool = [p for p in products if p["category"] == category and _eligible(p, profile, session)]
    # rank: concern matches, then real (non-demo) products, then cheaper first (unknown price counts as 0)
    scored = [(sum(c in p["concerns"] for c in concerns), 0 if p.get("demo") else 1,
               -(p.get("price_inr") or 0), p) for p in pool]
    if need_match:
        scored = [s for s in scored if s[0] > 0]
    if not scored:
        return None, 0
    best = max(scored, key=lambda s: s[:3])
    return best[3], best[0]


def _step(label, product, matched, concerns, profile):
    if product is None:
        return {"step": label, "product": None, "why": "No eligible product in your budget or skin type."}
    hit = [c for c in concerns if c in product["concerns"]]
    why = f"Targets {', '.join(hit)}." if matched and hit else f"Suits {profile['skin_type']} skin."
    return {
        "step": label,
        "product": {k: product.get(k) for k in ("id", "name", "brand", "price_inr", "image_url", "demo")},
        "why": why,
    }


def build_routine(profile: dict | None) -> dict:
    profile = normalise_profile(profile)
    concerns = profile["concerns"] or ["dryness"]  # default to basic hydration
    products = catalog.skincare()
    chosen = {"am": [], "pm": []}

    plan = {
        "am": [("Cleanser", "cleanser", False), ("Serum", "serum", False),
               ("Moisturizer", "moisturizer", False), ("Sunscreen", "sunscreen", False)],
        "pm": [("Cleanser", "cleanser", False), ("Treatment", "treatment", True),
               ("Moisturizer", "moisturizer", False)],
    }
    steps = {"am": [], "pm": []}
    used = {"am": [], "pm": []}
    for session in ("am", "pm"):
        for label, category, need_match in plan[session]:
            product, matched = _pick(products, category, session, profile, concerns, need_match)
            if product is None and need_match:
                continue  # optional step: skip when nothing fits
            steps[session].append(_step(label, product, matched, concerns, profile))
            if product:
                used[session].append(product)

    warnings = []
    notes = []
    for session in ("am", "pm"):
        for c in check_products(used[session]):
            warnings.append({**c, "session": session})
        notes += notes_for({a for p in used[session] for a in product_actives(p)})

    total = sum((p.get("price_inr") or 0) for session in used for p in used[session])
    return {
        "profile": profile,
        "am": steps["am"],
        "pm": steps["pm"],
        "warnings": warnings,
        "notes": sorted(set(notes)),
        "total_inr": total,
        "disclaimer": DISCLAIMER,
    }
