"""Rule-backed ingredient classification and conflict checking (cosmetic guidance, not medical advice)."""
import re

# Order matters for vitamin C: derivatives are checked first.
_PATTERNS = {
    "retinoid": r"\bretin\w*|\badapalene\b|\btretinoin\b",
    "aha": r"\baha\b|glycolic|lactic acid|mandelic",
    "bha": r"\bbha\b|salicylic",
    "benzoyl_peroxide": r"benzoyl",
    "niacinamide": r"niacinamide|vitamin b3",
    "azelaic": r"azelaic",
    "sunscreen": r"zinc oxide|titanium dioxide|avobenzone|octocrylene|homosalate|tinosorb|uvinul|\bspf\b|sunscreen",
    "hyaluronic": r"hyaluron",
    "ceramide": r"ceramide",
}
_VIT_C_DERIVATIVE = r"ascorbyl|ascorbate|ethyl ascorbic|ascorbic acid 2-glucoside"
_VIT_C = r"vitamin c|ascorbic acid|l-ascorbic"

LABELS = {
    "retinoid": "retinoid (retinol/retinal)",
    "aha": "AHA (glycolic/lactic/mandelic)",
    "bha": "BHA (salicylic acid)",
    "vitamin_c": "vitamin C (L-ascorbic acid)",
    "benzoyl_peroxide": "benzoyl peroxide",
    "niacinamide": "niacinamide",
    "azelaic": "azelaic acid",
}

# (a, b, severity, advice)
RULES = [
    ("retinoid", "aha", "avoid", "Both exfoliate and can irritate or over-dry skin. Use them on different nights."),
    ("retinoid", "bha", "avoid", "Stacking them raises the risk of irritation and peeling. Alternate nights."),
    ("retinoid", "benzoyl_peroxide", "avoid", "Benzoyl peroxide can irritate and may degrade some retinoids. Use it in the morning and the retinoid at night."),
    ("vitamin_c", "benzoyl_peroxide", "avoid", "Benzoyl peroxide can oxidise L-ascorbic acid and make it less effective. Keep them apart."),
    ("vitamin_c", "aha", "caution", "Both are acidic and can sting together. Vitamin C in the morning, acid at night."),
    ("vitamin_c", "bha", "caution", "Both are acidic and can sting together. Vitamin C in the morning, acid at night."),
    ("vitamin_c", "retinoid", "caution", "A common pairing when split: vitamin C in the morning, retinoid at night."),
    ("aha", "bha", "caution", "Using both can over-exfoliate. Alternate nights or pick one."),
]

ACTIVE_NOTES = {
    "retinoid": "Retinoid: start 2 nights a week, build up slowly, and wear sunscreen every morning.",
    "vitamin_c": "Vitamin C: use in the morning, followed by sunscreen.",
    "aha": "AHA: start 2 nights a week and avoid using it on the same night as a retinoid.",
    "bha": "BHA: start 2 to 3 nights a week; stop if skin feels tight or stings.",
}


def classify(text: str) -> set[str]:
    """Map one ingredient or product string to active classes."""
    t = text.lower()
    found = {name for name, pat in _PATTERNS.items() if re.search(pat, t)}
    if re.search(_VIT_C_DERIVATIVE, t):
        found.add("vitamin_c_derivative")
    elif re.search(_VIT_C, t):
        found.add("vitamin_c")
    return found


def classify_many(texts: list[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out |= classify(t)
    return out


def product_actives(product: dict) -> set[str]:
    return classify_many(product.get("ingredients", []))


def check_actives(actives: set[str]) -> list[dict]:
    conflicts = []
    for a, b, severity, advice in RULES:
        if a in actives and b in actives:
            conflicts.append({"between": [LABELS[a], LABELS[b]], "severity": severity, "advice": advice})
    return conflicts


def check_products(products: list[dict]) -> list[dict]:
    """Conflicts between different products used in the same session."""
    conflicts = []
    actives = [(p, product_actives(p)) for p in products]
    for a, b, severity, advice in RULES:
        for pa, aa in actives:
            for pb, ab in actives:
                if pa["id"] != pb["id"] and a in aa and b in ab:
                    conflicts.append({
                        "between": [pa["name"], pb["name"]],
                        "severity": severity,
                        "advice": advice,
                    })
    return conflicts


def notes_for(actives: set[str]) -> list[str]:
    return [ACTIVE_NOTES[a] for a in ("retinoid", "vitamin_c", "aha", "bha") if a in actives]
