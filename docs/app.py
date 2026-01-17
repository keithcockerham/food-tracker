from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, abort
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder=".", static_url_path="/static")

ROOT_DIR = Path(".").resolve()
DATA_DIR = (ROOT_DIR / "data").resolve()
DATA_DIR.mkdir(exist_ok=True)

INGREDIENTS_PATH = DATA_DIR / "ingredients.parquet"
RECIPES_PATH = DATA_DIR / "recipes.parquet"
CALENDAR_PATH = DATA_DIR / "calendar.parquet"
METRICS_PATH = DATA_DIR / "metrics.json"

NUTRIENT_COLS = ["calories", "fat", "protein", "carbs", "sodium", "fiber", "sugar"]

# LLM config

# NOTE: If you point OPENAI_BASE_URL at a local/OpenAI-compatible service,
# this app still expects OPENAI_API_KEY to be set. Use a dummy value if your
# provider ignores it.

load_dotenv("../.env")  
load_dotenv()           # also allow .env in the current working dir

LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip()

client = None

if LLM_PROVIDER in ("disabled", "none", "off", "false", "0"):
    client = None
elif LLM_PROVIDER in ("openai", "openai_compat", "openai-compatible"):
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is required when LLM_PROVIDER is enabled. "
            "Set it in .env or environment variables, or set LLM_PROVIDER=disabled."
        )
    if OPENAI_BASE_URL:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
else:
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Use 'openai' or 'disabled'."
    )

def _now():
    return datetime.utcnow()

def load_df(path: Path, columns=None) -> pd.DataFrame:
    if not path.exists():
        if columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=columns)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path)

def save_df(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records")

def compute_nutrients_for_entry(source_type: str, source_id: str, qty: float, unit: str):
    ingredients_df = load_df(INGREDIENTS_PATH)
    recipes_df = load_df(RECIPES_PATH)

    totals = {k: 0.0 for k in NUTRIENT_COLS}

    if source_type == "ingredient":
        row = ingredients_df.loc[ingredients_df.get("ingredient_id") == source_id]
        if row.empty:
            return totals
        row = row.iloc[0]
        base_amount = float(row.get("base_amount", 1.0) or 1.0)
        factor = qty / base_amount
        for col in NUTRIENT_COLS:
            totals[col] = factor * float(row.get(col, 0.0) or 0.0)
        return totals

    if source_type == "recipe":
        rec = recipes_df.loc[recipes_df.get("recipe_id") == source_id]
        if rec.empty:
            return totals
        rec = rec.iloc[0]
        servings_per_batch = float(rec.get("servings_per_batch", 1.0) or 1.0)

        try:
            ing_list = json.loads(rec.get("ingredients", "[]") or "[]")
        except json.JSONDecodeError:
            ing_list = []

        batch_totals = {k: 0.0 for k in NUTRIENT_COLS}
        for item in ing_list:
            ing_id = item.get("ingredient_id")
            ing_qty = float(item.get("qty", 0.0) or 0.0)

            ing_row = ingredients_df.loc[ingredients_df.get("ingredient_id") == ing_id]
            if ing_row.empty:
                continue
            ing_row = ing_row.iloc[0]
            base_amount = float(ing_row.get("base_amount", 1.0) or 1.0)
            factor_ing = ing_qty / base_amount
            for col in NUTRIENT_COLS:
                batch_totals[col] += factor_ing * float(ing_row.get(col, 0.0) or 0.0)

        factor_recipe = qty / servings_per_batch
        for col in NUTRIENT_COLS:
            totals[col] = batch_totals[col] * factor_recipe
        return totals

    return totals



# LLM Helpers + APIs (plan_bot.html)

def _safe_json(obj):
    """Best-effort JSON-safe conversion for context payloads."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {}

def _llm_chat(system: str, user: str, model: str = "gpt-4.1-mini", temperature: float = 0.7) -> str:
    """Minimal wrapper around the OpenAI chat API."""
    if client is None:
        raise RuntimeError("LLM is disabled (LLM_PROVIDER=disabled) or not configured.")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()

@app.post("/api/llm")
def api_llm():
    """
    Generic LLM endpoint used by plan_bot.html:
      - task: "helper_suggestions" | "substitution_suggestions" | "chat"
      - user_text: string (for chat)
      - context: dict (planner totals/targets/items)
    Returns: { "text": "..." }
    """
    if client is None:
        return jsonify({"error": "LLM is disabled or not configured"}), 503

    data = request.get_json(force=True) or {}
    task = (data.get("task") or "chat").strip()
    user_text = data.get("user_text") or ""
    context = _safe_json(data.get("context") or {})

    system = (
        "You are a practical nutrition/meal-planning assistant inside a Day Planner app.\n"
        "Be concise, specific, and actionable.\n"
        "Prefer lower-sodium options when reasonable.\n"
        "Do not invent nutrition numbers; use what the context provides and reason qualitatively.\n"
        "When giving suggestions, use bullet points."
    )

    if task == "helper_suggestions":
        prompt = (
            "Generate helper suggestions to improve the day plan toward the goals.\n"
            "Return 6-10 bullet points max.\n\n"
            f"Context (JSON):\n{json.dumps(context, indent=2)}"
        )
        text_out = _llm_chat(system, prompt, model="gpt-4.1-mini", temperature=0.7)
        return jsonify({"text": text_out})

    if task == "substitution_suggestions":
        prompt = (
            "Generate substitution ideas to improve the day plan toward the goals.\n"
            "Return 6-10 bullet points max. Include swap pairs like 'X -> Y'.\n\n"
            f"Context (JSON):\n{json.dumps(context, indent=2)}"
        )
        text_out = _llm_chat(system, prompt, model="gpt-4.1-mini", temperature=0.8)
        return jsonify({"text": text_out})

    # default: freeform chat with context
    prompt = (
        f"User message:\n{user_text}\n\n"
        f"Planner context (JSON):\n{json.dumps(context, indent=2)}"
    )
    text_out = _llm_chat(system, prompt, model="gpt-4.1-mini", temperature=0.7)
    return jsonify({"text": text_out})




@app.post("/api/llm_nutrition")
def api_llm_nutrition():
    """
    Used by add.html to suggest nutrition fields for a new ingredient.
    Input: { name: str, description: str }
    Output JSON:
      { base_amount, base_unit, calories, fat, protein, carbs, sodium, fiber, sugar }
    """
    if client is None:
        return jsonify({"error": "LLM is disabled or not configured"}), 503

    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()

    if not name:
        return jsonify({"error": "Missing name"}), 400

    system = (
        "You estimate nutrition for a single ingredient.\n"
        "Return ONLY valid JSON with keys:\n"
        "base_amount (number), base_unit (string), calories (number), fat (g), protein (g), carbs (g),\n"
        "sodium (mg), fiber (g), sugar (g).\n"
        "Use a reasonable default base_amount=1 and base_unit='serving' unless a better unit is obvious.\n"
        "If unsure, give conservative, typical estimates and keep sodium modest unless clearly salty.\n"
        "No extra text."
    )

    prompt = json.dumps({
        "name": name,
        "description": description,
        "notes": "Provide typical nutrition estimate per base amount/unit. JSON only."
    }, indent=2)

    raw = _llm_chat(system, prompt, model="gpt-4.1-mini", temperature=0.4)

    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            obj = json.loads(raw[start:end])
        except Exception:
            obj = None

    if not isinstance(obj, dict):

        obj = {
            "base_amount": 1,
            "base_unit": "serving",
            "calories": 0,
            "fat": 0,
            "protein": 0,
            "carbs": 0,
            "sodium": 0,
            "fiber": 0,
            "sugar": 0,
        }

    def num(v, default=0.0):
        try:
            if v is None or v == "":
                return default
            return float(v)
        except Exception:
            return default

    out = {
        "base_amount": num(obj.get("base_amount", 1), 1),
        "base_unit": str(obj.get("base_unit", "serving") or "serving"),
        "calories": num(obj.get("calories", 0)),
        "fat": num(obj.get("fat", 0)),
        "protein": num(obj.get("protein", 0)),
        "carbs": num(obj.get("carbs", 0)),
        "sodium": num(obj.get("sodium", 0)),
        "fiber": num(obj.get("fiber", 0)),
        "sugar": num(obj.get("sugar", 0)),
    }

    return jsonify(out)





def _default_metrics_payload() -> dict:
    return {
        "updated_at": _now().isoformat() + "Z",
        "targets": {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "fiber_g": 0,
            "sodium_mg": 0,
        },
    }


def _coerce_float(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


@app.get("/api/metrics")
def api_get_metrics():
    """Return metrics targets payload stored at data/metrics.json.

    Response shape:
      { updated_at: str, targets: { calories, protein_g, carbs_g, fat_g, fiber_g, sodium_mg } }

    If no file exists yet, returns a default payload (all zeros).
    """
    if not METRICS_PATH.exists():
        return jsonify(_default_metrics_payload())

    try:
        obj = json.loads(METRICS_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        return jsonify(_default_metrics_payload())

    if not isinstance(obj, dict):
        obj = {}

    targets = obj.get("targets") if isinstance(obj.get("targets"), dict) else {}

    out = {
        "updated_at": str(obj.get("updated_at") or _now().isoformat() + "Z"),
        "targets": {
            "calories": _coerce_float(targets.get("calories"), 0),
            "protein_g": _coerce_float(targets.get("protein_g"), 0),
            "carbs_g": _coerce_float(targets.get("carbs_g"), 0),
            "fat_g": _coerce_float(targets.get("fat_g"), 0),
            "fiber_g": _coerce_float(targets.get("fiber_g"), 0),
            "sodium_mg": _coerce_float(targets.get("sodium_mg"), 0),
        },
    }
    return jsonify(out)


@app.post("/api/metrics")
def api_save_metrics():
    """Persist daily target metrics to data/metrics.json.

    Accepts either:
      1) full payload: { updated_at, targets: {...} }
      2) bare targets: { calories, protein_g, ... }
    """
    data = request.get_json(force=True) or {}

    if isinstance(data.get("targets"), dict):
        targets_in = data.get("targets") or {}
    else:
        targets_in = data

    payload = {
        "updated_at": _now().isoformat() + "Z",
        "targets": {
            "calories": _coerce_float(targets_in.get("calories"), 0),
            "protein_g": _coerce_float(targets_in.get("protein_g"), 0),
            "carbs_g": _coerce_float(targets_in.get("carbs_g"), 0),
            "fat_g": _coerce_float(targets_in.get("fat_g"), 0),
            "fiber_g": _coerce_float(targets_in.get("fiber_g"), 0),
            "sodium_mg": _coerce_float(targets_in.get("sodium_mg"), 0),
        },
    }

    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify({"status": "ok", "path": str(METRICS_PATH), **payload})

@app.post("/api/meal_genie")
def api_meal_genie():
    """
    Meal Genie endpoint used by plan_bot.html.
    Input:
      - goal: dict (targets like calories/protein/etc)
      - requirements: string (positive constraints, e.g., high-protein, low-sodium)
    Output:
      - menu_text: string
      - shopping_list_text: string
      - structured: dict (best-effort JSON extraction)

    TEMP VERSION NOTE:
      This version requires cooking instructions for breakfast/lunch/dinner under the ingredient list.
    """
    data = request.get_json(force=True) or {}
    goal = _safe_json(data.get("goal") or {})
    reqs = (data.get("requirements") or "").strip()

    if client is None:
        return jsonify({"error": "LLM is disabled or not configured"}), 503

    system = (
        "You are Meal Genie. You generate a RANDOM 3-day menu and a consolidated shopping list."
        "Aim to match the daily goals. Respect positive requirements."
        "Keep meals realistic for a normal household. Repeats are allowed."
        "OUTPUT MUST BE ONLY ONE VALID JSON OBJECT. No extra text, no markdown."
        "JSON KEYS REQUIRED: menu_text, shopping_list_text, structured"
        "MENU_TEXT FORMAT REQUIREMENTS (human-readable):"

        "- 3 days labeled Day 1, Day 2, Day 3"
        "- Each day includes Breakfast, Lunch, Dinner, Snack 1, Snack 2"
        "- For Breakfast/Lunch/Dinner: list ingredients with quantities AND then include cooking instructions."
        "  Example:"
        "  Breakfast: Egg White Omelet (1 serving):"
        "    - 4 egg whites"
        "    - 1 cup spinach"
        "    Instructions: Heat pan, saute spinach, add egg whites, cook until set."
        "- Snacks may be no-cook; include prep instructions if any (optional)."

        "SHOPPING_LIST_TEXT REQUIREMENTS:"
        "- Consolidated totals across ALL 3 days"
        "- Group by category: Produce, Protein/Dairy, Pantry/Grains, Canned/Jarred, Spices/Other, Beverages"
        "- Each line includes a total amount and unit"

        "STRUCTURED JSON REQUIREMENTS:"
        "structured must include:"
        "- days: ["
        "    { day: 1, day_totals: macros, meals: {"
        "        breakfast: { name, servings, ingredients:[{item, amount, unit}], instructions, macros },"
        "        lunch:     { ... },"
        "        dinner:    { ... },"
        "        snack1:    { name, servings, ingredients:[...], instructions(optional), macros },"
        "        snack2:    { ... }"
        "    }}" 

        ""
        "  ]"
        "- shopping_list: { category_name: [ { item, total_amount, unit, notes } ] }"
        "MACRO REQUIREMENTS (LLM MUST PROVIDE):"
        "- Provide macro estimates for EACH MEAL and DAY TOTALS."
        "- Use exact keys: calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg"
        "- Estimates can be approximate but must be internally consistent (day_totals ~= sum of meals)."
        "- Do not return zeros."
    )

    payload = {
        "goal": goal,
        "positive_requirements": reqs,
        "macro_keys": [
            "calories_kcal","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg"
        ],
        "categories": ["Produce","Protein/Dairy","Pantry/Grains","Canned/Jarred","Spices/Other","Beverages"],
        "notes": "Return only JSON with keys menu_text, shopping_list_text, structured. Breakfast/lunch/dinner must include instructions under ingredients in menu_text and in structured.meals.<meal>.instructions."
    }

    raw = _llm_chat(system, json.dumps(payload, indent=2), model="gpt-4.1", temperature=0.9)

    menu_text = ""
    shopping_list_text = ""
    structured = None

    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            obj = json.loads(raw[start:end])
        except Exception:
            obj = None

    if isinstance(obj, dict):
        menu_text = (obj.get("menu_text") or "").strip()
        shopping_list_text = (obj.get("shopping_list_text") or "").strip()
        structured = obj.get("structured", None)

    # if JSON failed, at least return the raw text in menu_text
    if not menu_text:
        menu_text = (raw or "").strip()
    if not shopping_list_text:
        shopping_list_text = ""

    return jsonify({
        "menu_text": menu_text,
        "shopping_list_text": shopping_list_text,
        "structured": structured
    })


@app.get("/api/ingredients")
def api_get_ingredients():
    df = load_df(INGREDIENTS_PATH)
    return df.to_json(orient="records")

@app.post("/api/ingredients")
def api_add_ingredient():
    data = request.json or {}
    df = load_df(INGREDIENTS_PATH)

    ingredient_id = data.get("ingredient_id") or f"ing_{uuid.uuid4().hex[:8]}"
    now = _now()

    record = {
        "ingredient_id": ingredient_id,
        "name": data["name"],
        "description": data.get("description", ""),
        "base_amount": float(data.get("base_amount", 1.0) or 1.0),
        "base_unit": data.get("base_unit", "serving") or "serving",
        "calories": float(data.get("calories", 0) or 0),
        "fat": float(data.get("fat", 0) or 0),
        "protein": float(data.get("protein", 0) or 0),
        "carbs": float(data.get("carbs", 0) or 0),
        "sodium": float(data.get("sodium", 0) or 0),
        "fiber": float(data.get("fiber", 0) or 0),
        "sugar": float(data.get("sugar", 0) or 0),
        "created_at": data.get("created_at", now),
        "updated_at": now,
    }

    if df.empty:
        df = pd.DataFrame([record])
    else:
        if "ingredient_id" in df.columns:
            df = df[df["ingredient_id"] != ingredient_id]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    save_df(df, INGREDIENTS_PATH)
    return jsonify({"status": "ok", "ingredient_id": ingredient_id})


@app.get("/api/recipes")
def api_get_recipes():
    df = load_df(RECIPES_PATH)
    return df.to_json(orient="records")

@app.post("/api/recipes")
def api_add_recipe():
    data = request.json or {}
    df = load_df(RECIPES_PATH)

    recipe_id = data.get("recipe_id") or f"rec_{uuid.uuid4().hex[:8]}"
    now = _now()

    record = {
        "recipe_id": recipe_id,
        "name": data["name"],
        "description": data.get("description", ""),
        "instructions": data.get("instructions", ""),
        "notes": data.get("notes", ""),
        "rating": data.get("rating", None),
        "servings_per_batch": float(data.get("servings_per_batch", 1.0) or 1.0),
        "ingredients": json.dumps(data.get("ingredients", [])),
        "created_at": data.get("created_at", now),
        "updated_at": now,
    }

    if df.empty:
        df = pd.DataFrame([record])
    else:
        if "recipe_id" in df.columns:
            df = df[df["recipe_id"] != recipe_id]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    save_df(df, RECIPES_PATH)
    return jsonify({"status": "ok", "recipe_id": recipe_id})


@app.get("/api/calendar")
def api_get_calendar():
    df = load_df(CALENDAR_PATH)
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.post("/api/calendar")
def api_add_calendar_entry():
    data = request.json or {}
    df = load_df(CALENDAR_PATH)

    entry_id = f"cal_{uuid.uuid4().hex[:8]}"
    dt = data.get("date")
    if not dt:
        return jsonify({"error": "date is required"}), 400

    source_type = data.get("source_type")
    source_id = data.get("source_id")
    if not source_type or not source_id:
        return jsonify({"error": "source_type and source_id are required"}), 400

    qty = float(data.get("qty", 1.0) or 1.0)
    unit = data.get("unit", "serving") or "serving"
    timestamp = data.get("timestamp")  

    nutrients = compute_nutrients_for_entry(source_type, source_id, qty, unit)

    record = {
        "entry_id": entry_id,
        "date": dt,
        "timestamp": timestamp,
        "source_type": source_type,
        "source_id": source_id,
        "qty": qty,
        "unit": unit,
    }
    record.update(nutrients)

    if df.empty:
        df = pd.DataFrame([record])
    else:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    save_df(df, CALENDAR_PATH)
    return jsonify({"status": "ok", "entry_id": entry_id})


@app.put("/api/calendar/<entry_id>")
def api_update_calendar_entry(entry_id):
    data = request.json or {}
    df = load_df(CALENDAR_PATH)
    if df.empty:
        return jsonify({"error": "calendar is empty"}), 404

    if "entry_id" not in df.columns:
        return jsonify({"error": "calendar data missing entry_id column"}), 500

    idx = df.index[df["entry_id"] == entry_id].tolist()
    if not idx:
        return jsonify({"error": f"entry_id not found: {entry_id}"}), 404
    i = idx[0]

    date = data.get("date", df.at[i, "date"])
    timestamp = data.get("timestamp", df.at[i, "timestamp"] if "timestamp" in df.columns else None)
    source_type = data.get("source_type", df.at[i, "source_type"])
    source_id = data.get("source_id", df.at[i, "source_id"])
    qty = float(data.get("qty", df.at[i, "qty"]))
    unit = data.get("unit", df.at[i, "unit"] if "unit" in df.columns else "serving") or "serving"

    if not date:
        return jsonify({"error": "date is required"}), 400
    if not source_type or not source_id:
        return jsonify({"error": "source_type and source_id are required"}), 400

    nutrients = compute_nutrients_for_entry(source_type, source_id, qty, unit)

    df.at[i, "date"] = date
    df.at[i, "timestamp"] = timestamp
    df.at[i, "source_type"] = source_type
    df.at[i, "source_id"] = source_id
    df.at[i, "qty"] = qty
    df.at[i, "unit"] = unit


    for k, v in nutrients.items():
        df.at[i, k] = v

    save_df(df, CALENDAR_PATH)
    return jsonify({"status": "ok", "entry_id": entry_id})


@app.delete("/api/calendar/<entry_id>")
def api_delete_calendar_entry(entry_id):
    df = load_df(CALENDAR_PATH)
    if df.empty:
        return jsonify({"status": "ok", "deleted": 0})

    if "entry_id" not in df.columns:
        return jsonify({"error": "calendar data missing entry_id column"}), 500

    before = len(df)
    df = df[df["entry_id"] != entry_id].copy()
    deleted = before - len(df)

    save_df(df, CALENDAR_PATH)
    return jsonify({"status": "ok", "deleted": deleted, "entry_id": entry_id})



import tools.reports_api_parquet as reports_api
reports_api.PARQUET_PATH = CALENDAR_PATH
app.register_blueprint(reports_api.bp)

ALLOWED_SUFFIXES = {
    ".html", ".css", ".js",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".json", ".map",
}

@app.get("/")
def home():
    idx = ROOT_DIR / "index.html"
    if idx.exists():
        return send_from_directory(str(ROOT_DIR), "index.html")
    abort(404)

@app.get("/<path:filename>")
def serve_any(filename: str):
    if filename.startswith("api/") or filename.startswith("static/"):
        abort(404)

    requested = (ROOT_DIR / filename).resolve()
    if not str(requested).startswith(str(ROOT_DIR)):
        abort(404)

    if requested.suffix.lower() not in ALLOWED_SUFFIXES:
        abort(404)

    if not requested.exists() or not requested.is_file():
        abort(404)

    rel = requested.relative_to(ROOT_DIR).as_posix()
    return send_from_directory(str(ROOT_DIR), rel)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
