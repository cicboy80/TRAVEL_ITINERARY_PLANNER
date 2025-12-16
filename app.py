# app.py 

import os
import re
import json
import time
import traceback
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from crewai import Agent, Task, Crew

from tools.google_maps_tool import GoogleMapsTool
from tools.route_planner_tool import RoutePlannerTool
from tools.weather_tool import WeatherTool
from tools.semantic_ranking_tool import SemanticRankingTool

from models.itinerary_model import ItineraryModel
from utils.date_utils import expand_dates

# ----------------------------
# Env
# ----------------------------
load_dotenv()
os.environ["LITELLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

print("🔍 GOOGLE_MAPS_API_KEY found:", bool(os.getenv("GOOGLE_MAPS_API_KEY")))
print("🔍 OPENAI_API_KEY found:", bool(os.getenv("OPENAI_API_KEY")))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY")
if not os.getenv("GOOGLE_MAPS_API_KEY"):
    raise ValueError("Missing GOOGLE_MAPS_API_KEY")

# ----------------------------
# Tools (Python-owned)
# ----------------------------
maps_tool = GoogleMapsTool()
weather_tool = WeatherTool()
route_tool = RoutePlannerTool()
semantic_tool = SemanticRankingTool()

# ----------------------------
# Helpers (preferences)
# ----------------------------
def preference_terms(preferences: str) -> List[str]:
    """Free-text -> list of preference terms (no hardcoding like 'bars')."""
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
    return out[:12]

def extract_activity_terms(preferences: str, max_terms: int = 8) -> List[str]:
    """
    Extract “activity-like” terms from preferences.
    We avoid duplicating obvious meal intent tokens, but do NOT hardcode venues like bars.
    """
    if not preferences:
        return []
    raw: List[str] = []
    for part in preferences.replace(";", ",").split(","):
        term = part.strip()
        if term:
            raw.append(term)

    # skip meal/food intent tokens (still not hardcoding venues)
    skip_tokens = ("food", "restaurant", "cuisine", "breakfast", "lunch", "dinner", "eat", "eating")
    out: List[str] = []
    for t in raw:
        tl = t.lower()
        if any(tok in tl for tok in skip_tokens):
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_terms:
            break
    return out

def to_iso_date(d: Any) -> str:
    if isinstance(d, str):
        s = d.replace("/", "-")
        return datetime.fromisoformat(s).date().isoformat()
    if hasattr(d, "date"):
        return d.date().isoformat()
    raise ValueError(f"Unsupported date value: {d!r}")

# ----------------------------
# Helpers (data shaping)
# ----------------------------
def compact_places(places_by_cat: Dict[str, List[Dict[str, Any]]], per_cat: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    """
    Keep only fields we need + cap count per category to keep planner context compact.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cat, items in (places_by_cat or {}).items():
        if not isinstance(items, list):
            continue

        # Sort roughly by "quality": rating desc, then user_ratings_total desc
        def score(x: Dict[str, Any]) -> Tuple[float, int]:
            r = x.get("rating") or 0.0
            n = x.get("user_ratings_total") or 0
            return (float(r), int(n))

        items_sorted = sorted(items, key=score, reverse=True)[:per_cat]

        cleaned: List[Dict[str, Any]] = []
        for x in items_sorted:
            addr = (x.get("address") or "").strip()
            if not addr:
                continue
            cleaned.append(
                {
                    "name": (x.get("name") or "").strip(),
                    "address": addr,
                    "category": (x.get("category") or cat).strip(),
                    "rating": x.get("rating"),
                    "user_ratings_total": x.get("user_ratings_total"),
                    "place_id": x.get("place_id"),
                }
            )
        out[cat] = cleaned
    return out

def flatten_candidates(compact: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Build candidates list for semantic ranking tool.
    Tool expects: name, category, description (we provide lightweight deterministic description).
    """
    cands: List[Dict[str, Any]] = []
    for cat, items in compact.items():
        for it in items:
            nm = it.get("name")
            addr = it.get("address")
            if not nm or not addr:
                continue
            cands.append(
                {
                    "name": nm,
                    "category": cat,
                    "description": f"{cat} option at {addr}",
                    "address": addr,
                    "rating": it.get("rating"),
                    "place_id": it.get("place_id"),
                }
            )
    # de-dupe by address (preserve order)
    seen = set()
    out = []
    for x in cands:
        k = x["address"].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def semantic_rank(preferences: str, candidates: List[Dict[str, Any]], top_k: int = 30) -> List[Dict[str, Any]]:
    if not preferences or not candidates:
        return candidates[:top_k]
    ranked = semantic_tool.run(
        user_preferences=preferences,
        candidates=[{"name": c["name"], "category": c["category"], "description": c["description"]} for c in candidates],
        top_k=top_k,
        distance_weight=0.0,
    )
    # ranked returns list[dict] with name/category/description/semantic_score
    # reattach address/place_id/rating deterministically by name+category match, fallback to name match
    by_key = {(c["name"], c["category"]): c for c in candidates}
    by_name = {}
    for c in candidates:
        by_name.setdefault(c["name"], c)

    out = []
    for r in ranked:
        k = (r.get("name"), r.get("category"))
        c = by_key.get(k) or by_name.get(r.get("name"))
        if not c:
            continue
        merged = dict(c)
        merged["semantic_score"] = r.get("semantic_score")
        out.append(merged)
    return out[:top_k]

def split_ranked_by_meals(
    ranked: List[Dict[str, Any]],
    meal_cats: Tuple[str, str, str] = ("breakfast", "lunch", "dinner"),
    k_meal: int = 4,
    k_activity: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    meals = {m: [] for m in meal_cats}
    activities: List[Dict[str, Any]] = []

    for x in ranked:
        cat = (x.get("category") or "").lower().strip()
        if cat in meals and len(meals[cat]) < k_meal:
            meals[cat].append(x)
        elif cat not in meal_cats:
            activities.append(x)

    # diversify activities by category a bit
    picked: List[Dict[str, Any]] = []
    per_cat_cap = 2
    per_cat_count: Dict[str, int] = {}

    for x in activities:
        cat = (x.get("category") or "").lower().strip() or "other"
        if per_cat_count.get(cat, 0) >= per_cat_cap:
            continue
        picked.append(x)
        per_cat_count[cat] = per_cat_count.get(cat, 0) + 1
        if len(picked) >= k_activity:
            break

    return {
        "breakfast": meals["breakfast"],
        "lunch": meals["lunch"],
        "dinner": meals["dinner"],
        "activities": picked,
    }

def build_allowed_locations(bundle: Dict[str, List[Dict[str, Any]]], max_destinations: int = 9) -> List[str]:
    """
    RoutePlannerTool NxN caps to 10 stops (origin + 9 destinations).
    So we choose a compact allowed set for the planner to pick from.
    """
    selected: List[str] = []

    def take(items: List[Dict[str, Any]], n: int):
        nonlocal selected
        for it in items[:n]:
            addr = (it.get("address") or "").strip()
            if addr and addr not in selected:
                selected.append(addr)

    # meals first (structure), then activities
    take(bundle.get("breakfast", []), 2)
    take(bundle.get("lunch", []), 2)
    take(bundle.get("dinner", []), 2)
    take(bundle.get("activities", []), 5)

    return selected[:max_destinations]

def place_lookup(compact: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Address -> place metadata."""
    out: Dict[str, Dict[str, Any]] = {}
    for cat, items in compact.items():
        for it in items:
            addr = (it.get("address") or "").strip()
            if not addr:
                continue
            out[addr] = it
    return out

def maps_search_url(address: str, place_id: Optional[str] = None) -> str:
    q = urllib.parse.quote_plus(address)
    if place_id:
        # Google Maps "search" url supports query_place_id
        pid = urllib.parse.quote_plus(place_id)
        return f"https://www.google.com/maps/search/?api=1&query={q}&query_place_id={pid}"
    return f"https://www.google.com/maps/search/?api=1&query={q}"

# ----------------------------
# Helper functions
# ----------------------------

def choose_mode_for_leg(
    allowed_modes: List[str],
    matrix: Dict[str, Any],
    i: int,
    j: int,
) -> Optional[str]:
    """
    Mode choice rule:
    - if walking allowed AND walking distance <= 2.0km -> walking
    - else if public_transport allowed -> public_transport
    - else if cycling allowed -> cycling
    - else -> driving
    Fallback: first mode with non-null duration.
    """
    def get_dist(mode: str) -> Optional[float]:
        try:
            return matrix[mode]["distance_km"][i][j]
        except Exception:
            return None

    def get_dur(mode: str) -> Optional[float]:
        try:
            return matrix[mode]["duration_min"][i][j]
        except Exception:
            return None

    if "walking" in allowed_modes:
        d = get_dist("walking")
        if d is not None and d <= 2.0:
            return "walking"

    if "public_transport" in allowed_modes and get_dur("public_transport") is not None:
        return "public_transport"

    if "cycling" in allowed_modes and get_dur("cycling") is not None:
        return "cycling"

    if "driving" in allowed_modes and get_dur("driving") is not None:
        return "driving"

    for m in ("walking", "public_transport", "cycling", "driving"):
        if m in allowed_modes and get_dur(m) is not None:
            return m

    return None


def compute_leg_metrics_from_matrix(
    routes: Dict[str, Any],
    allowed_modes: List[str],
    from_loc: str,
    to_loc: str,
) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    """
    Returns: (mode, distance_km, duration_minutes_int) using NxN matrix.
    """
    stops = routes.get("stops") or []
    matrix = routes.get("matrix") or {}
    if not stops or not matrix:
        return (None, None, None)

    idx = {stops[k]: k for k in range(len(stops))}
    if from_loc not in idx or to_loc not in idx:
        return (None, None, None)

    i, j = idx[from_loc], idx[to_loc]
    mode = choose_mode_for_leg(allowed_modes, matrix, i, j)
    if not mode:
        return (None, None, None)

    try:
        d = matrix[mode]["distance_km"][i][j]
        t = matrix[mode]["duration_min"][i][j]
    except Exception:
        return (None, None, None)

    if d is None or t is None:
        return (mode, None, None)

    return (mode, float(d), int(round(float(t))))

def compute_leg_metrics_pairwise(
    route_tool: RoutePlannerTool,
    allowed_modes: List[str],
    from_loc: str,
    to_loc: str,
) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    """
    Pairwise fallback: calls RoutePlannerTool for a single leg (from -> to).
    Handles RoutePlannerTool returning either a list or a dict.
    """

    try:
        out = route_tool.run(
            origin=from_loc,
            destinations=[to_loc],
            modes=allowed_modes,
            max_results=1,
            return_matrix=False,
        )
    except Exception:
        return (None, None, None)

    # ✅ Normalize output into `best` (a dict for the best leg/route)
    if isinstance(out, dict):
        # common patterns
        if "routes" in out and isinstance(out["routes"], list) and out["routes"]:
            best = out["routes"][0]
        elif "results" in out and isinstance(out["results"], list) and out["results"]:
            best = out["results"][0]
        else:
            best = out
    elif isinstance(out, list) and out:
        best = out[0]
    else:
        return (None, None, None)

    # ✅ Extract fields (support a couple of common key names)
    mode = best.get("mode") or best.get("travel_mode")
    dist = best.get("distance_km") or best.get("distance")  # if your tool uses "distance"
    dur = best.get("duration_min") or best.get("duration_minutes") or best.get("duration")

    # ✅ Coerce types safely
    try:
        dist_km = float(dist) if dist is not None else None
    except Exception:
        dist_km = None

    try:
        dur_min = int(round(float(dur))) if dur is not None else None
    except Exception:
        dur_min = None

    return (mode, dist_km, dur_min)

def extract_unique_locations_in_order(itinerary: ItineraryModel) -> List[str]:
    """
    Unique activity locations in the order they appear across days.
    """
    seen = set()
    out: List[str] = []
    for day in itinerary.days:
        for act in day.activities:
            loc = (act.location or "").strip()
            if loc and loc not in seen:
                seen.add(loc)
                out.append(loc)
    return out


def postprocess_itinerary(
    itinerary: ItineraryModel,
    route_tool: RoutePlannerTool,
    allowed_modes: List[str],
    origin: str,
    addr_to_meta: Dict[str, Dict[str, Any]],
    prefer_single_matrix: bool = True,
    nxn_destination_cap: int = 9,  # origin + 9 = 10 stops
) -> ItineraryModel:
    """
    Option A:
    - Planner selects activities first
    - Then we compute travel metrics
      - If <= cap unique locations: single NxN matrix call
      - Else: pairwise per-leg calls (still cached)

    Always fills:
    - act.map_url
    - act.travel_mode, act.distance_from_prev, act.duration_minutes
    - itinerary.total_distance_km
    """
    total_km = 0.0

    # Fill map_url first (deterministic)
    for day in itinerary.days:
        for act in day.activities:
            meta = addr_to_meta.get(act.location)
            act.map_url = maps_search_url(act.location, meta.get("place_id") if meta else None)

    # Decide routing strategy
    unique_locs = extract_unique_locations_in_order(itinerary)
    use_matrix = prefer_single_matrix and (len(unique_locs) <= nxn_destination_cap)

    routes: Dict[str, Any] = {}
    if use_matrix:
        destinations = unique_locs[:nxn_destination_cap]
        routes = route_tool.run(
            origin=origin,
            destinations=destinations,
            modes=allowed_modes,
            max_results=len(destinations),
            return_matrix=True,
        )

    # Now compute legs day-by-day in itinerary order
    for day in itinerary.days:
        prev_loc = origin
        for act in day.activities:
            if use_matrix:
                mode, dist_km, dur_min = compute_leg_metrics_from_matrix(routes, allowed_modes, prev_loc, act.location)
                # if matrix misses a leg (shouldn’t, but safe), fall back pairwise
                if (mode is None) or (dist_km is None) or (dur_min is None):
                    mode2, dist2, dur2 = compute_leg_metrics_pairwise(route_tool, allowed_modes, prev_loc, act.location)
                    # only overwrite if pairwise gave something better
                    if mode2 is not None: mode = mode2
                    if dist_km is None: dist_km = dist2
                    if dur_min is None: dur_min = dur2
            else:
                mode, dist_km, dur_min = compute_leg_metrics_pairwise(route_tool, allowed_modes, prev_loc, act.location)

            act.travel_mode = mode
            act.distance_from_prev = dist_km
            act.duration_minutes = dur_min

            if dist_km is not None:
                total_km += dist_km

            prev_loc = act.location

    itinerary.total_distance_km = round(total_km, 2)

    # Note if we had to use pairwise due to size
    if not use_matrix and len(unique_locs) > nxn_destination_cap:
        msg = (
            f"Routing note: itinerary contains {len(unique_locs)} unique locations; "
            f"computed travel metrics using per-leg routing (matrix cap is {nxn_destination_cap} destinations)."
        )
        itinerary.notes = (itinerary.notes + "\n" + msg) if itinerary.notes else msg

    return itinerary


# ----------------------------
# Agents (LLM-only)
# ----------------------------
planner_agent = Agent(
    role="Senior Travel Agent",
    goal="Create a trip plan strictly from CONTEXT_JSON. Never invent places.",
    backstory="You are a senior travel agent who plans realistic schedules using only provided options.",
    llm="gpt-5-mini",
    temperature=0.25,
    verbose=False,
    tools=[],  # IMPORTANT: no tools in planner (Python did all tools)
)

writer_agent = Agent(
    role="Itinerary Writer Agent",
    goal="Write Markdown from the provided itinerary JSON without inventing new places.",
    backstory="You write clear, engaging itineraries that stick to the provided structured plan.",
    llm="gpt-4o-mini",
    temperature=0.3,
    verbose=False,
)

# ----------------------------
# Core logic
# ----------------------------
def generate_itinerary(location, start_date, end_date, preferences, transport_modes):
    t0 = time.time()
    print("✅ Button clicked:", location, start_date, end_date, transport_modes, preferences)

    try:
        if not location or not str(location).strip():
            return "### ❌ Error\nPlease enter a destination."

        # Normalize dates
        start_date_iso = to_iso_date(start_date)
        end_date_iso = to_iso_date(end_date)
        days_count, date_list = expand_dates(start_date_iso, end_date_iso)

        if isinstance(transport_modes, str):
            transport_modes = [transport_modes]
        transport_modes = [m for m in (transport_modes or []) if m]
        if not transport_modes:
            transport_modes = ["walking"]

        pref_terms = preference_terms(preferences or "")
        activity_terms = extract_activity_terms(preferences or "")

        # --------------------
        # 1) Python: Maps (once)
        # --------------------
        # cuisine_preferences is optional and your maps tool expects a dict keyed by meals,
        # but we do not "guess cuisines" from preferences; we keep it None for neutrality.
        places_raw = maps_tool.run(
            location=location,
            activities=activity_terms if (preferences and activity_terms) else None,
            cuisine_preferences=None,
            max_results_per_query=20,
        )

        # Make places compact + deterministic
        places_compact = compact_places(places_raw, per_cat=6)
        candidates = flatten_candidates(places_compact)
        addr_to_meta = place_lookup(places_compact)

        # --------------------
        # 2) Python: Weather (once)
        # --------------------
        weather_raw = weather_tool.run(
            city=location,
            start_date=start_date_iso,
            end_date=end_date_iso,
        )

        # Convert weather list -> date map
        weather_by_date: Dict[str, Dict[str, Any]] = {}
        for row in weather_raw.get("forecasts", []) or []:
            d = row.get("date")
            if not d:
                continue
            weather_by_date[d] = {
                "temp_max": row.get("temp_max"),
                "temp_min": row.get("temp_min"),
                "precipitation_mm": row.get("precipitation_mm"),
            }

        # --------------------
        # 3) Python: Semantic pre-rank (once)
        # --------------------
        ranked = semantic_rank(preferences or "", candidates, top_k=30)

        # split into meal lists + activity list (still preference-driven)
        bundle = split_ranked_by_meals(ranked, k_meal=4, k_activity=10)

        # --------------------
        # 5) LLM: Planner (no tools) — compact deterministic context
        # --------------------
        context_json = {
            "location": location,
            "start_date": start_date_iso,
            "end_date": end_date_iso,
            "trip_duration_days": days_count,
            "preferences": preferences or "",
            "transport_modes": transport_modes,
            "places": {
                "breakfast": bundle["breakfast"],
                "lunch": bundle["lunch"],
                "dinner": bundle["dinner"],
                "activities": bundle["activities"],
            },
            "weather": weather_by_date,
            "RULES": [
                "Use ONLY the provided places lists. Never invent new places.",
                "Each activity.location MUST be exactly one of the provided place addresses (verbatim).",
                "You must produce exactly trip_duration_days day-plans, matching the date range.",
                "Keep each day within 08:00–22:00.",
                "Use weather to bias indoor vs outdoor choices (rain -> museums/indoor).",
                "Do not worry about distance/time fields; Python will compute travel metrics after planning.",
                "Still output travel_mode/distance_from_prev/duration_minutes fields (can be null).",
                # IMPORTANT for Option A + matrix efficiency (even though we have pairwise fallback):
                "Prefer <= 9 unique locations total across the itinerary to keep routing efficient.",
            ],
        }

        planning_task = Task(
            description=(
                "Build an itinerary ONLY from the provided CONTEXT_JSON. Never invent places.\n\n"
                f"CONTEXT_JSON={json.dumps(context_json, ensure_ascii=False)}\n\n"
                "Output MUST match ItineraryModel exactly.\n"
            ),
            expected_output="A JSON itinerary matching ItineraryModel exactly.",
            agent=planner_agent,
            output_pydantic=ItineraryModel,
        )

        planner_crew = Crew(
            agents=[planner_agent],
            tasks=[planning_task],
            verbose=True,
        )

        _ = planner_crew.kickoff()

        # Extract the Pydantic object (CrewAI stores in task.output)
        planned = getattr(planning_task, "output", None)
        planned_obj = getattr(planned, "pydantic", None) or getattr(planned, "output_pydantic", None)

        if planned_obj is None:
            # fallback: try parse raw json
            raw = getattr(planned, "raw", None) or str(planned)
            planned_obj = ItineraryModel.model_validate_json(raw)

        itinerary: ItineraryModel = planned_obj

        # --------------------
        # 6) Python: overwrite travel fields deterministically (THE FIX)
        # --------------------
        itinerary = postprocess_itinerary(
            itinerary=itinerary,
            route_tool=route_tool,
            allowed_modes=transport_modes,
            origin=location,
            addr_to_meta=addr_to_meta,
            prefer_single_matrix=True,   # will use NxN when possible
            nxn_destination_cap=9,       # origin + 9 = 10 stops
        )

        itinerary_json = itinerary.model_dump()

        # --------------------
        # 7) LLM: Writer — Markdown from corrected JSON
        # --------------------
        writing_task = Task(
            description=(
                "Write Markdown from ITINERARY_JSON. Do NOT invent places.\n\n"
                f"**Trip Dates:** {start_date_iso} → {end_date_iso}\n\n"
                f"ITINERARY_JSON={json.dumps(itinerary_json, ensure_ascii=False)}\n\n"
                "Formatting rules:\n"
                "- Use a header per day.\n"
                "- Bold timestamps.\n"
                "- Include rating inline when present (⭐ 4.7).\n"
                "- For each day: include a travel line for every activity *except the first activity of that day*.\n"
                "- Travel line format (exactly): *Travel from previous: {duration_minutes} min, {distance_from_prev} km ({travel_mode})*\n"
                "- Only describe places in the JSON.\n"
            ),
            expected_output="Markdown itinerary.",
            agent=writer_agent,
        )

        writer_crew = Crew(
            agents=[writer_agent],
            tasks=[writing_task],
            verbose=True,
        )

        result = writer_crew.kickoff()
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

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧭 AI-Powered Travel Itinerary Planner")
    gr.Markdown("Plan optimized, weather-aware trips based on your preferences")

    with gr.Row():
        location = gr.Textbox(label="Destination", placeholder="e.g. Venice, Italy")
        transport_modes = gr.CheckboxGroup(
            ["walking", "public_transport", "driving", "cycling"],
            label="Transport modes",
            value=["walking"],
            info="Select one or multiple modes for route optimization",
        )

    with gr.Row():
        start_date = gr.DateTime(label="Start Date", include_time=False, type="datetime")
        end_date = gr.DateTime(label="End Date", include_time=False, type="datetime")

    preferences = gr.Textbox(label="Your Preferences", placeholder="art, local food, museums, relaxed pace, avoid queues")

    generate_btn = gr.Button("📝 Generate Itinerary")
    itinerary_markdown = gr.Markdown(label="🔖 Your Personalized Itinerary")

    generate_btn.click(
        fn=generate_itinerary,
        inputs=[location, start_date, end_date, preferences, transport_modes],
        outputs=[itinerary_markdown],
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