"""Ask Hydra: a Claude tool-calling agent over the app's own data, with an offline fallback."""
import json
import os

from . import catalog
from .config import MODEL_OVERRIDE, PROVIDER_OVERRIDE, PROVIDERS
from .ingredients import check_actives, classify, classify_many, product_actives
from .routine import build_routine, normalise_profile

SYSTEM = """You are Hydra, a friendly skincare and makeup assistant inside the Hydra.ai app, talking to Gen Z users.
Rules:
- Give cosmetic guidance only, never medical diagnosis. For severe, painful, or persistent skin problems, suggest a dermatologist.
- Use the tools for facts: the user's saved profile and scan, routines, ingredient conflicts, and products. Only recommend products returned by search_products or build_routine; never invent product names or prices.
- If you lack information (skin type, concerns), ask one short question.
- Keep replies short, warm, and practical. Prices are in INR. Use plain text without markdown tables."""

TOOLS = [
    {"name": "get_user_profile",
     "description": "Get the user's saved skin type, concerns, budget, and latest face-scan result (tone, undertone).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "build_routine",
     "description": "Build the user's AM/PM skincare routine from their saved profile, with conflict warnings.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "check_ingredient_conflicts",
     "description": "Check whether skincare ingredients or active types (e.g. 'retinol', 'glycolic acid', 'vitamin C') conflict when used in the same session.",
     "input_schema": {"type": "object",
                      "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
                      "required": ["ingredients"]}},
    {"name": "search_products",
     "description": "Search the product catalog. kind 'skincare' categories: cleanser, serum, treatment, moisturizer, sunscreen. kind 'makeup' categories: lipstick, blush, foundation. concern is one of acne, pigmentation, aging, dullness, redness, dryness.",
     "input_schema": {"type": "object",
                      "properties": {"kind": {"type": "string", "enum": ["skincare", "makeup"]},
                                     "category": {"type": "string"},
                                     "concern": {"type": "string"},
                                     "max_price_inr": {"type": "number"}},
                      "required": ["kind"]}},
    {"name": "get_game_state",
     "description": "Get the user's Hydra Points, level, streak, badges, and today's check-ins.",
     "input_schema": {"type": "object", "properties": {}}},
]


def _to_openai_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


OPENAI_TOOLS = _to_openai_tools(TOOLS)


def _ollama_running(base: str) -> bool:
    try:
        import requests
        return requests.get(base.replace("/v1", "") + "/api/tags", timeout=0.6).ok
    except Exception:
        return False


def resolve_provider() -> str | None:
    """Pick the chat provider: explicit HYDRA_PROVIDER, else the first free option that is configured."""
    if PROVIDER_OVERRIDE:
        return None if PROVIDER_OVERRIDE == "offline" else PROVIDER_OVERRIDE
    for name in ("gemini", "groq", "anthropic"):
        if os.getenv(PROVIDERS[name]["key_env"] or ""):
            return name
    return "ollama" if _ollama_running(PROVIDERS["ollama"]["base"]) else None


class HydraAgent:
    def __init__(self, store, client=None, model: str | None = None, provider: str | None = "auto", http_post=None):
        self.store = store
        self.client = client          # anthropic-style client (tests inject a fake)
        self._http_post = http_post   # OpenAI-compatible POST (tests inject a fake)
        self.provider = resolve_provider() if provider == "auto" else provider
        if client is not None:
            self.provider = "anthropic"
        cfg = PROVIDERS.get(self.provider or "", {})
        self.model = model or MODEL_OVERRIDE or cfg.get("model", "")
        if self.provider == "anthropic" and self.client is None:
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                self.provider = None

    @property
    def available(self) -> bool:
        return self.provider is not None

    def _post(self, payload: dict) -> dict:
        if self._http_post:
            return self._http_post(payload)
        import requests
        cfg = PROVIDERS[self.provider]
        headers = {"Content-Type": "application/json"}
        key = os.getenv(cfg["key_env"] or "")
        if cfg["key_env"] and not key:
            raise RuntimeError(f"{cfg['key_env']} is not set in .env")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(cfg["base"] + "/chat/completions", headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    # ---- tools -----------------------------------------------------
    def _run_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "get_user_profile":
                return {"profile": normalise_profile(self.store.get_profile()), "latest_scan": self.store.latest_scan()}
            if name == "build_routine":
                return build_routine(self.store.get_profile())
            if name == "check_ingredient_conflicts":
                actives = classify_many([str(i) for i in args.get("ingredients", [])])
                conflicts = check_actives(actives)
                return {"recognised_actives": sorted(actives), "conflicts": conflicts,
                        "ok": not any(c["severity"] == "avoid" for c in conflicts)}
            if name == "search_products":
                return {"results": self._search(args)}
            if name == "get_game_state":
                return self.store.state()
            return {"error": f"unknown tool {name}"}
        except Exception as exc:
            return {"error": str(exc)}

    def _search(self, args: dict) -> list[dict]:
        kind = args.get("kind", "skincare")
        items = catalog.skincare() if kind == "skincare" else catalog.makeup()
        cat, concern, cap = args.get("category"), args.get("concern"), args.get("max_price_inr")
        out = []
        for p in items:
            if cat and p["category"] != cat:
                continue
            if concern and kind == "skincare" and concern not in p["concerns"]:
                continue
            if cap and p.get("price_inr") is not None and p["price_inr"] > cap:
                continue
            row = {k: p.get(k) for k in ("id", "category", "name", "brand", "price_inr")}
            row["source"] = p.get("source", "demo catalog")
            if kind == "skincare":
                row["actives"] = sorted(product_actives(p))
                row["good_for"] = p["concerns"]
                row["skin_types"] = p["skin_types"]
            else:
                row["undertone"] = p["undertone"]
            out.append(row)
        return out[:8]

    # ---- chat ------------------------------------------------------
    @staticmethod
    def _clean_history(history, message: str) -> list[dict]:
        msgs: list[dict] = []
        for h in (history or [])[-10:]:
            role, content = h.get("role"), h.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                if msgs and msgs[-1]["role"] == role:
                    msgs[-1]["content"] += "\n" + content
                else:
                    msgs.append({"role": role, "content": content})
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] += "\n" + message
        else:
            msgs.append({"role": "user", "content": message})
        return msgs

    def chat(self, message: str, history=None) -> dict:
        message = (message or "").strip()
        if not message:
            return {"reply": "Ask me anything about your skin, routine, or products.", "tools_used": [], "mode": "offline"}
        if not self.available:
            return self._offline(message)
        msgs = self._clean_history(history, message)
        used: list[str] = []
        try:
            if self.provider == "anthropic":
                return self._chat_anthropic(msgs, used)
            return self._chat_openai(msgs, used)
        except Exception as exc:
            return {"reply": f"The {self.provider} chat model is unavailable right now ({type(exc).__name__}: {exc}). "
                             "Check your key or that Ollama is running. Offline answers still work for ingredient questions.",
                    "tools_used": used, "mode": "error"}

    def _chat_openai(self, msgs, used):
        """OpenAI-compatible tool loop (Gemini, Groq, Ollama)."""
        convo = [{"role": "system", "content": SYSTEM}] + msgs
        for _ in range(6):
            data = self._post({"model": self.model, "messages": convo, "tools": OPENAI_TOOLS, "temperature": 0.4})
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if calls:
                for i, c in enumerate(calls):
                    c["id"] = c.get("id") or f"call_{i}"
                convo.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for c in calls:
                    fn = c["function"]
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args or "{}")
                        except ValueError:
                            args = {}
                    used.append(fn["name"])
                    convo.append({"role": "tool", "tool_call_id": c["id"], "name": fn["name"],
                                  "content": json.dumps(self._run_tool(fn["name"], args), default=str)})
                continue
            return {"reply": (msg.get("content") or "").strip() or "Hmm, I have nothing to add there.",
                    "tools_used": used, "mode": self.provider}
        return {"reply": "I got a bit tangled up. Could you rephrase that?", "tools_used": used, "mode": self.provider}

    def _chat_anthropic(self, msgs, used):
        for _ in range(6):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1000, system=SYSTEM, tools=TOOLS, messages=msgs)
            if resp.stop_reason == "tool_use":
                msgs.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        used.append(block.name)
                        out = self._run_tool(block.name, block.input or {})
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": json.dumps(out, default=str)})
                msgs.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return {"reply": text or "Hmm, I have nothing to add there.", "tools_used": used, "mode": "anthropic"}
        return {"reply": "I got a bit tangled up. Could you rephrase that?", "tools_used": used, "mode": "anthropic"}

    # ---- offline fallback (no API key) -----------------------------
    def _offline(self, message: str) -> dict:
        actives = classify(message)
        notice = "(Offline mode: add a free GEMINI_API_KEY to .env, or run Ollama, for full chat.)"
        if len(actives) >= 2:
            conflicts = check_actives(actives)
            if conflicts:
                lines = [f"- {' + '.join(c['between'])} [{c['severity']}]: {c['advice']}" for c in conflicts]
                return {"reply": "Here is what I found:\n" + "\n".join(lines) + "\n" + notice,
                        "tools_used": ["check_ingredient_conflicts"], "mode": "offline"}
            return {"reply": "I don't see a known conflict between those, but patch-test new actives first.\n" + notice,
                    "tools_used": ["check_ingredient_conflicts"], "mode": "offline"}
        if "routine" in message.lower():
            r = build_routine(self.store.get_profile())
            fmt = lambda steps: ", ".join(f"{s['step']}: {s['product']['name']}" for s in steps if s["product"])
            return {"reply": f"AM: {fmt(r['am'])}\nPM: {fmt(r['pm'])}\n{notice}",
                    "tools_used": ["build_routine"], "mode": "offline"}
        return {"reply": "I can check ingredient conflicts (try: 'can I use retinol with glycolic acid?') "
                         "or show your routine (ask for 'my routine'). " + notice,
                "tools_used": [], "mode": "offline"}
