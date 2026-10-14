"""Core tests (no webcam, no network). Run:  python -m pytest -v   or   python tests/test_core.py"""
import math
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra import catalog
from hydra.agent import HydraAgent
from hydra.game import GameStore
from hydra.ingredients import check_actives, check_products, classify
from hydra import products_live
from hydra.routine import build_routine
from hydra.skin import CHEEK_LANDMARKS, analyze_skin, match_foundations, recommend_by_fit
from hydra.tryon import INNER_LIPS, LEFT_EDGE, OUTER_LIPS, RIGHT_EDGE, apply_blush, apply_lipstick


def fake_landmarks():
    lm = np.full((478, 2), 0.5, dtype=np.float32)
    for n, i in enumerate(OUTER_LIPS):
        t = 2 * math.pi * n / len(OUTER_LIPS)
        lm[i] = (0.5 + 0.10 * math.cos(t), 0.75 + 0.04 * math.sin(t))
    for n, i in enumerate(INNER_LIPS):
        t = 2 * math.pi * n / len(INNER_LIPS)
        lm[i] = (0.5 + 0.06 * math.cos(t), 0.75 + 0.015 * math.sin(t))
    pts = [(.35, .6), (.36, .62), (.33, .6), (.34, .58), (.65, .6), (.64, .62), (.67, .6), (.66, .58)]
    for i, p in zip(CHEEK_LANDMARKS, pts):
        lm[i] = p
    lm[205], lm[425] = (0.35, 0.6), (0.65, 0.6)
    lm[LEFT_EDGE], lm[RIGHT_EDGE] = (0.2, 0.5), (0.8, 0.5)
    return lm


def solid(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return np.full((480, 640, 3), (b, g, r), np.uint8)


def test_skin_tone_and_matching():
    lm, foundations = fake_landmarks(), catalog.makeup_in("foundation")
    expected = {"#F3D5BF": "Porcelain 100", "#C28E63": "Caramel 300", "#8F5E3C": "Cocoa 400", "#55331F": "Mahogany 500"}
    depths = []
    for hex_color, name in expected.items():
        skin = analyze_skin(solid(hex_color), lm)
        assert match_foundations(skin, foundations, 1)[0]["name"] == name
        depths.append(skin["depth"])
    assert depths == ["light", "medium", "deep", "deep"]
    assert recommend_by_fit(analyze_skin(solid("#8F5E3C"), lm), catalog.makeup_in("lipstick"), 1)


def test_tryon_changes_only_target_pixels():
    lm, frame = fake_landmarks(), solid("#C28E63")
    out = apply_blush(apply_lipstick(frame, lm, "#8B2C4A"), lm, "#E58A9A")
    changed = (out != frame).any(axis=2)
    assert out.shape == frame.shape and 1000 < changed.sum() < frame.shape[0] * frame.shape[1] * 0.2
    assert (out[0, 0] == frame[0, 0]).all()


def test_ingredient_rules():
    assert classify("Sodium Ascorbyl Phosphate") == {"vitamin_c_derivative"}
    assert "vitamin_c" in classify("Ascorbic Acid") and "vitamin_c" not in classify("Ethyl Ascorbic Acid")
    assert any(c["severity"] == "avoid" for c in check_actives(classify("retinol") | classify("glycolic acid")))
    assert not check_actives(classify("niacinamide") | classify("hyaluronic acid"))
    prods = [p for p in catalog.skincare() if p["id"] in ("T01", "T03")]
    assert check_products(prods)


def test_routine_safety():
    sens = build_routine({"skin_type": "sensitive", "concerns": ["aging", "acne"]})
    names = [s["product"]["id"] for s in sens["am"] + sens["pm"] if s["product"]]
    assert not {"T01", "T02", "T03", "S02"} & set(names)  # no harsh actives for sensitive skin
    norm = build_routine({"skin_type": "normal", "concerns": ["aging"], "avoid_retinoids": True})
    assert "T01" not in [s["product"]["id"] for s in norm["pm"] if s["product"]]
    full = build_routine({"skin_type": "oily", "concerns": ["acne", "pigmentation"]})
    assert full["am"] and full["pm"] and not any(w["severity"] == "avoid" for w in full["warnings"])
    cheap = build_routine({"skin_type": "oily", "budget_inr": 100})
    assert all(s["product"] is None for s in cheap["am"])


def test_game_points_streak_unlock():
    with tempfile.TemporaryDirectory() as d:
        g = GameStore(Path(d) / "t.db")
        day = date(2026, 10, 1)
        assert g.checkin("am", day)["awarded"] == 10
        assert g.checkin("am", day)["awarded"] == 0  # no double dipping
        assert g.checkin("pm", day)["awarded"] == 15  # +5 full-day bonus
        for i in range(1, 7):
            r = g.checkin("am", day + timedelta(days=i))
        assert r["state"]["streak"] == 7 and r["awarded"] == 60  # +50 week streak
        assert g.record_scan({"tone": "x"}, day) == 50 and g.record_scan({"tone": "y"}, day + timedelta(days=2)) == 0
        assert g.record_scan({"tone": "z"}, day + timedelta(days=9)) == 20
        assert not g.unlock("L08", 99999)["ok"]
        before = g.state(day)["points"]
        assert g.unlock("L06", 60)["ok"] and g.state(day)["points"] == before - 60 and "L06" in g.unlocked()
        g.set_profile({"skin_type": "oily"})
        assert g.get_profile()["skin_type"] == "oily" and g.latest_scan()["tone"] == "z"


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create: first asks for a tool, then answers."""
    def __init__(self):
        self.calls, self.messages = 0, NS(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        if self.calls == 1:
            block = NS(type="tool_use", id="t1", name="check_ingredient_conflicts",
                       input={"ingredients": ["retinol", "glycolic acid"]})
            return NS(stop_reason="tool_use", content=[block])
        last = kw["messages"][-1]["content"][0]
        assert last["type"] == "tool_result" and "avoid" in last["content"]
        return NS(stop_reason="end_turn", content=[NS(type="text", text="Use them on different nights.")])


def test_agent_tool_loop_and_offline():
    with tempfile.TemporaryDirectory() as d:
        store = GameStore(Path(d) / "t.db")
        r = HydraAgent(store, client=FakeClient()).chat("retinol with glycolic?", [{"role": "assistant", "content": "hi"}])
        assert r["mode"] == "anthropic" and r["tools_used"] == ["check_ingredient_conflicts"] and "nights" in r["reply"]
        a = HydraAgent(store, provider=None)
        off = a.chat("Can I use retinol with glycolic acid?")
        assert off["mode"] == "offline" and "avoid" in off["reply"]
        assert a.chat("show my routine")["tools_used"] == ["build_routine"]


OBF_SAMPLE = {"products": [
    {"code": "111", "product_name": "Niacinamide 10% Serum", "brands": "TestBrand, Other",
     "ingredients_text": "Aqua, Niacinamide, Zinc PCA, Glycerin.", "categories": "Serums",
     "image_front_small_url": "https://images.example.com/a.jpg"},
    {"code": "222", "product_name": "Mystery Thing", "brands": "X", "ingredients_text": ""},
    {"product_name": "No barcode"},
]}


def test_open_beauty_facts_normalise_and_import():
    found = products_live.search_live("niacinamide", http_get=lambda url, params=None: OBF_SAMPLE)
    assert [p["id"] for p in found] == ["obf:111", "obf:222"]
    p = found[0]
    assert p["brand"] == "TestBrand" and p["category"] == "serum" and "niacinamide" in str(p["concerns"]) or "acne" in p["concerns"]
    assert p["ingredients"][:2] == ["Aqua", "Niacinamide"] and p["price_inr"] is None and p["image_url"].startswith("https://")
    assert found[1]["ingredients"] == []
    one = products_live.get_by_barcode("111", http_get=lambda url, params=None: {"product": OBF_SAMPLE["products"][0]})
    assert one["id"] == "obf:111"
    assert products_live.get_by_barcode("0", http_get=lambda url, params=None: {}) is None


def test_custom_product_validation_and_routine_prefers_real():
    try:
        products_live.from_user("X", "B", "serum", "", None)
        assert False, "empty ingredients should fail"
    except ValueError:
        pass
    mine = products_live.from_user("My Retinol", "Brand", "treatment", "Water, Retinol, Squalane", None)
    assert mine["sessions"] == ["pm"] and "aging" in mine["concerns"] and mine["demo"] is False
    catalog.load_extra([mine])
    try:
        r = build_routine({"skin_type": "normal", "concerns": ["aging"]})
        assert r["pm"][1]["product"]["id"] == mine["id"]  # real product beats the demo one on a tie
        assert r["total_inr"] >= 0
        sens = build_routine({"skin_type": "sensitive", "concerns": ["aging"]})
        assert mine["id"] not in [s["product"]["id"] for s in sens["pm"] if s["product"]]
    finally:
        catalog.load_extra([])


def test_free_provider_tool_loop_and_resolution():
    calls = []

    def fake_post(payload):
        calls.append(payload)
        if len(calls) == 1:
            assert payload["tools"][0]["type"] == "function" and payload["messages"][0]["role"] == "system"
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "", "type": "function", "function": {"name": "check_ingredient_conflicts",
                 "arguments": '{"ingredients": ["retinol", "glycolic acid"]}'}}]}}]}
        tool_msg = payload["messages"][-1]
        assert tool_msg["role"] == "tool" and "avoid" in tool_msg["content"] and tool_msg["tool_call_id"]
        return {"choices": [{"message": {"content": "Alternate nights."}}]}

    with tempfile.TemporaryDirectory() as d:
        store = GameStore(Path(d) / "t.db")
        agent = HydraAgent(store, provider="gemini", http_post=fake_post)
        r = agent.chat("retinol with glycolic?")
        assert r["mode"] == "gemini" and r["reply"] == "Alternate nights." and r["tools_used"] == ["check_ingredient_conflicts"]
        assert agent.model  # provider default model is set
        broken = HydraAgent(store, provider="ollama", http_post=lambda p: (_ for _ in ()).throw(RuntimeError("down")))
        assert broken.chat("hi")["mode"] == "error"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
