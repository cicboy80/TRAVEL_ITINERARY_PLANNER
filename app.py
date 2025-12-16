import os
import gradio as gr
import traceback
import json
import time
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# =========================
# ENV
# =========================
load_dotenv()

print("🔍 GOOGLE_MAPS_API_KEY found:", bool(os.getenv("GOOGLE_MAPS_API_KEY")))
print("🔍 OPENAI_API_KEY found:", bool(os.getenv("OPENAI_API_KEY")))

os.environ["LITELLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

# Disable tracing + avoid prompts (harmless if your CrewAI version ignores these)
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TRACES_PROMPT"] = "true"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY")
if not os.getenv("GOOGLE_MAPS_API_KEY"):
    print("⚠️ Warning: Missing GOOGLE_MAPS_API_KEY")

# =========================
# IMPORTS (local)
# =========================
from tools.google_maps_tool import GoogleMapsTool
from tools.route_planner_tool import RoutePlannerTool
from tools.weather_tool import WeatherTool
from tools.semantic_ranking_tool import SemanticRankingTool
from models.itinerary_model import ItineraryModel
from utils.date_utils import expand_dates

# =========================
# TOOLS (Python-called phase)
# =========================
maps_tool = GoogleMapsTool()
weather_tool = WeatherTool()
route_tool = RoutePlannerTool()
semantic_ranking_tool = SemanticRankingTool()

# =========================
# AGENTS (LLM-only phase)
# =========================
planner_agent = Agent(
    role="Senior Travel Agent",
    goal=(
        "Design a balanced daily itinerary using ONLY the provided CONTEXT_JSON. "
        "Never invent places not present in the context."
    ),
    backstory=(
        "A senior travel agent expert at building efficient, preference-matching itineraries with realistic timings."
    ),
    tools=[semantic_ranking_tool],
    llm="gpt-5-mini",
    temperature=0.3,
    reasoning=False,
    verbose=False,
)

writer_agent = Agent(
    role="Itinerary Writer Agent",
    goal="Turn the structured itinerary JSON into engaging Markdown without inventing new places.",
    backstory="Professional travel writer, accurate and non-fictional.",
    llm="gpt-4o-mini",
    temperature=0.3,
    verbose=False,
)

# =========================
# HELPERS
# =========================
def normalize_date(d: Any) -> str:
    if isinstance(d, str):
        d = d.replace("/", "-")
        return datetime.fromisoformat(d).date().isoformat()
    if hasattr(d, "date"):
        return d.date().isoformat()
    raise ValueError(f"Unsupported date value: {d!r}")

def preference_terms(preferences: str, cap: int = 12) -> List[str]:
    """Turn free-text preferences into a clean list of query intents (no hardcoding of specific venues)."""
    if not preferences:
        return []
    parts = re.split(r"[,\n;/|]+", preferences)
    terms = [p.strip() for p in parts if p.strip()]
    seen = set()
    out = []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:cap]

def split_terms_for_maps(terms: List[str]) -> Tuple[List[str], Optional[Dict[str, str]]]:
    """
    Derive:
      - activities: list[str] used as dynamic categories for GoogleMapsTool
      - cuisine_preferences: optional mapping meal->query-text
    No hardcoded "bars/craft beer" — we just route whatever the user wrote into queries.
    """
    if not terms:
        return [], None

    meal_tokens = ("breakfast", "lunch", "dinner", "restaurant", "food", "cuisine")
    activities = [t for t in terms if not any(tok in t.lower() for tok in meal_tokens)]
    activities = activities[:8]

    foodish_tokens = (
        "food", "cuisine", "restaurant", "vegan", "vegetarian", "gluten",
        "wine", "beer", "coffee", "bakery", "pasta", "pizza", "seafood"
    )
    food_terms = [t for t in terms if any(tok in t.lower() for tok in foodish_tokens)]
    cuisine_text = ", ".join(food_terms).strip() if food_terms else None

    cuisine_preferences = None
    if cuisine_text:
        # Your GoogleMapsTool expects Dict[str,str] keyed by meal category
        cuisine_preferences = {"breakfast": cuisine_text, "lunch": cuisine_text, "dinner": cuisine_text}

    return activities, cuisine_preferences

def pick_address(item: Any, fallback_city: str) -> Optional[str]:
    if isinstance(item, dict):
        addr = (item.get("formatted_address") or item.get("address") or "").strip()
        if addr:
            return addr
        name = (item.get("name") or "").strip()
        return f"{name}, {fallback_city}".strip(", ") if name else None
    s = str(item).strip()
    return s if s else None

def compact_places(maps_data: Dict[str, Any], per_cat: int = 6, max_cats: int = 14) -> Dict[str, List[Dict[str, Any]]]:
    """
    Output: { category: [ {name,address,rating,place_id,category,lat,lng,user_ratings_total}, ... ] }
    Slim + stable for planner context.
    """
    if not isinstance(maps_data, dict):
        return {}

    # Normalize if tool returns {"restaurants": {...}, "landmarks":[...], ...}
    if "restaurants" in maps_data and isinstance(maps_data["restaurants"], dict):
        for k, v in maps_data["restaurants"].items():
            if k not in maps_data:
                maps_data[k] = v

    out: Dict[str, List[Dict[str, Any]]] = {}
    cat_count = 0
    for cat, items in maps_data.items():
        if cat_count >= max_cats:
            break
        if not isinstance(items, list):
            continue
        slim: List[Dict[str, Any]] = []
        for r in items[:per_cat]:
            if not isinstance(r, dict):
                continue
            slim.append({
                "name": r.get("name"),
                "address": r.get("address") or r.get("formatted_address"),
                "rating": r.get("rating"),
                "place_id": r.get("place_id"),
                "category": r.get("category") or cat,
                "lat": r.get("lat"),
                "lng": r.get("lng"),
                "user_ratings_total": r.get("user_ratings_total"),
            })
        if slim:
            out[cat] = slim
            cat_count += 1
    return out

def build_destinations_for_routes(places_by_cat: Dict[str, List[Dict[str, Any]]], location: str, cap: int = 12) -> List[str]:
    """
    Deterministic: meals first, then preference-driven categories (in dict order).
    Slightly >10 helps with “craft beer” appearing in the routeable stops if retrieved.
    """
    dests: List[str] = []

    def add_from(cat: str, n: int):
        for item in places_by_cat.get(cat, [])[:n]:
            addr = pick_address(item, location)
            if addr:
                dests.append(addr)

    # Meals
    add_from("breakfast", 2)
    add_from("lunch", 2)
    add_from("dinner", 2)

    # Then everything else
    for cat in places_by_cat.keys():
        if cat in ("breakfast", "lunch", "dinner"):
            continue
        add_from(cat, 2)
        if len(dests) >= cap:
            break

    # De-dupe preserve order
    seen = set()
    out: List[str] = []
    for d in dests:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out[:cap]

def slim_weather_for_context(weather_data: Any) -> Any:
    """
    Your WeatherTool returns:
      { city, latitude, longitude, start_date, end_date, forecast_days, forecasts:[...] }
    We shrink it to {YYYY-MM-DD:{temp_max,temp_min,precipitation_mm}} for lower token cost.
    """
    if isinstance(weather_data, dict) and isinstance(weather_data.get("forecasts"), list):
        slim: Dict[str, Dict[str, Any]] = {}
        for f in weather_data["forecasts"]:
            if not isinstance(f, dict):
                continue
            dt = f.get("date")
            if not dt:
                continue
            slim[dt] = {
                "temp_max": f.get("temp_max"),
                "temp_min": f.get("temp_min"),
                "precipitation_mm": f.get("precipitation_mm"),
            }
        return slim
    return weather_data

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)

# =========================
# CORE LOGIC (Fix 2)
# =========================
def generate_itinerary(location, start_date, end_date, preferences, transport_modes):
    t0 = time.time()
    print("✅ Button clicked:", location, start_date, end_date, transport_modes, preferences)

    try:
        # ---- normalize dates ----
        start_date = normalize_date(start_date)
        end_date = normalize_date(end_date)
        trip_duration_days, _ = expand_dates(start_date, end_date)

        # ---- normalize modes ----
        if isinstance(transport_modes, str):
            transport_modes = [transport_modes]
        transport_modes = transport_modes or ["walking"]

        # =========================
        # (A) Python: GOOGLE MAPS (once)
        # =========================
        terms = preference_terms(preferences)
        activities, cuisine_preferences = split_terms_for_maps(terms)

        maps_data = maps_tool.run(
            location=location,
            activities=activities if activities else None,
            cuisine_preferences=cuisine_preferences,
            max_results_per_query=20,
        )

        places_by_cat = compact_places(maps_data, per_cat=6, max_cats=14)
        if not places_by_cat:
            raise RuntimeError("Google Maps tool returned no usable places.")

        # =========================
        # (B) Python: WEATHER (once) — EXACT SIGNATURE FOR YOUR WeatherTool
        # =========================
        weather_data = weather_tool.run(
            city=location,
            start_date=start_date,
            end_date=end_date,
        )
        weather_slim = slim_weather_for_context(weather_data)

        # =========================
        # (C) Python: ROUTE MATRIX (once)
        # =========================
        destinations = build_destinations_for_routes(places_by_cat, location, cap=12)
        if not destinations:
            raise RuntimeError("No destinations built for routing.")

        route_data = route_tool.run(
            origin=location,
            destinations=destinations,
            modes=transport_modes,
            max_results=12,
            return_matrix=True,
        )

        # Slim route data for context
        route_slim = {
            "stops": route_data.get("stops"),
            "modes_requested": route_data.get("modes_requested"),
            "matrix": route_data.get("matrix"),
        }

        if not route_slim["stops"] or not route_slim["matrix"]:
            raise RuntimeError("Route tool did not return usable stops/matrix.")

        # =========================
        # (D) Compact deterministic context for the planner
        # =========================
        context = {
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "trip_duration_days": trip_duration_days,
            "preferences": preferences,
            "transport_modes": transport_modes,
            "places": places_by_cat,
            "weather": weather_slim,
            "routes": route_slim,
        }
        context_json = safe_json_dumps(context)

        # =========================
        # (E) LLM: Planner + Writer
        # =========================
        planning_task = Task(
            description=(
                "Build an itinerary ONLY from the provided CONTEXT_JSON.\n"
                "Never invent places.\n\n"
                f"CONTEXT_JSON={context_json}\n\n"
                "Rules:\n"
                "- Output MUST match ItineraryModel exactly.\n"
                "- Each activity.location MUST be exactly one of routes.stops (verbatim).\n"
                "- Use routes.matrix for consecutive travel times.\n"
                "- Mode choice:\n"
                "  • if walking allowed and walking distance <= 2.0km -> walking\n"
                "  • else if public_transport allowed -> public_transport\n"
                "  • else if cycling allowed -> cycling\n"
                "  • else -> driving\n"
                "- Keep day within 08:00–22:00.\n"
                "- Use weather to bias indoor/outdoor.\n"
                "- Ensure variety across days.\n"
                "- Include: name, category, start_time, end_time, location, rating (if any), reasoning, "
                "weather_forecast (short), distance_from_prev, duration_minutes, travel_mode.\n"
                f"- transport_modes must equal: {safe_json_dumps(transport_modes)}\n"
            ),
            expected_output="Structured JSON itinerary matching ItineraryModel.",
            agent=planner_agent,
            output_pydantic=ItineraryModel,
        )

        writing_task = Task(
            description=(
                "Write Markdown from the itinerary JSON produced by the planner.\n"
                "Do NOT invent places.\n"
                f"At the top include: **Trip Dates:** {start_date} → {end_date}\n"
                "For each activity include rating inline when present and a travel line when present.\n"
                "Use headers per day and bold timestamps."
            ),
            expected_output="Markdown itinerary.",
            agent=writer_agent,
            context=[planning_task],
        )

        crew = Crew(
            agents=[planner_agent, writer_agent],
            tasks=[planning_task, writing_task],
            verbose=True,
        )

        result = crew.kickoff(inputs={})
        markdown_itinerary = (
            result if isinstance(result, str)
            else getattr(result, "raw", None) or str(result)
        )

        print(f"✅ Done in {time.time() - t0:.1f}s")
        return markdown_itinerary

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return f"### ❌ Error\n```text\n{tb}\n```"

# =========================
# GRADIO UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧭 AI-Powered Travel Itinerary Planner")
    gr.Markdown("Plan optimized, weather-aware trips powered by multi-agent reasoning 🌍")

    with gr.Row():
        location = gr.Textbox(label="Destination", placeholder="e.g. Venice, Italy")
        transport_modes = gr.CheckboxGroup(
            ["walking", "public_transport", "driving", "cycling"],
            label="transport_modes",
            value=["walking"],
            info="Select one or multiple modes for route optimization"
        )

    with gr.Row():
        start_date = gr.DateTime(label="Start Date", include_time=False, type="datetime")
        end_date = gr.DateTime(label="End Date", include_time=False, type="datetime")

    preferences = gr.Textbox(label="Your Preferences", placeholder="art, local food, relaxed pace, avoid queues")

    generate_btn = gr.Button("📝 Generate Itinerary")
    itinerary_markdown = gr.Markdown(label="🔖 Your Personalized Itinerary")

    generate_btn.click(
        fn=generate_itinerary,
        inputs=[location, start_date, end_date, preferences, transport_modes],
        outputs=[itinerary_markdown]
    )

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        max_threads=1,
        ssr_mode=False,
    )