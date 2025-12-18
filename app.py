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
    skip_tokens = ("food", "restaurant", "cuisine", "breakfast", "lunch", "dinner", "eat", "eating",
                   "vegan", "vegetarian", "gluten-free", "gluten free")
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
            addr = (x.get("address") or x.get("formatted_address") or x.get("vicinity") or "").strip()
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
                    "types": x.get("types") or [],
                    "price_level": x.get("price_level"),
                    "business_status": x.get("business_status"),
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
            types = ", ".join(it.get("types") or [])
            price = it.get("price_level")
            rating = it.get("rating")
            reviews = it.get("user_ratings_total")
            desc = (
                f"{cat}. rating={rating} reviews={reviews} "
                f"price_level={price} types={types}. Address={addr}"
            )

            cands.append(
                {
                    "name": nm,
                    "category": cat,
                    "description": desc,
                    "address": addr,
                    "rating": rating,
                    "place_id": it.get("place_id"),
                    "user_ratings_total": reviews,
                    "price_level": price,
                    "types": it.get("types") or [],
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
        candidates=candidates,
        top_k=top_k,
        distance_weight=0.0,
    )

    return ranked[:top_k] if isinstance(ranked, list) else []

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

def build_meal_hints(preferences: str) -> Dict[str, Optional[str]]:
    """
    Turns user preferences into short text hints for Google Places queries:
      query = f"{hint} {meal} in {location}"
    Returns: {"breakfast": "...", "lunch": "...", "dinner": "..."} with None if no hint.
    """
    if not preferences or not str(preferences).strip():
        return {"breakfast": None, "lunch": None, "dinner": None}

    # basic tokenization for comma-separated / free text
    raw = re.split(r"[,\n;/|]+", preferences.lower())
    tokens = {t.strip().replace("_", "-") for t in raw if t.strip()}

    # canonical sets
    DIETARY = {
        "vegan": "vegan",
        "vegetarian": "vegetarian",
        "gluten-free": "gluten free",
        "gluten free": "gluten free",
    }

    CUISINES = {
        "italian": "italian",
        "thai": "thai",
        "japanese": "japanese",
        "chinese": "chinese",
        "french": "french",
        "seafood": "seafood",
        "local": "local",
    }

    QUALITY = {
        "michelin-star": "michelin star",
        "michelin star": "michelin star",
        "michelin": "michelin star",
    }

    # extract (order matters; keep short)
    dietary = []
    for key in ("vegan", "vegetarian", "gluten-free", "gluten free"):
        if key in tokens:
            dietary.append(DIETARY[key])

    cuisines = []
    for key in ("local", "italian", "thai", "japanese", "chinese", "french", "seafood"):
        if key in tokens:
            cuisines.append(CUISINES[key])

    michelin = None
    for key in ("michelin-star", "michelin star", "michelin"):
        if key in tokens:
            michelin = QUALITY[key]
            break

    # simple conflict guard: if vegan/vegetarian, drop seafood keyword
    if any(x in dietary for x in ("vegan", "vegetarian")):
        cuisines = [c for c in cuisines if c != "seafood"]

    # cap to keep queries tight
    dietary = dietary[:2]            # e.g. "vegan gluten free"
    cuisines = cuisines[:2]          # e.g. "local italian"

    def hint_for(meal: str) -> Optional[str]:
        parts = []
        parts += dietary
        parts += cuisines

        # Michelin is most relevant for lunch/dinner
        if michelin and meal in ("lunch", "dinner"):
            parts.append(michelin)

        # keep it compact
        hint = " ".join(parts).strip()
        return hint if hint else None

    return {
        "breakfast": hint_for("breakfast"),
        "lunch": hint_for("lunch"),
        "dinner": hint_for("dinner"),
    }

def extract_vibes(preferences: str, max_terms: int = 3) -> List[str]:
    if not preferences or not str(preferences).strip():
        return []

    raw = re.split(r"[,\n;/|]+", preferences.lower())
    tokens = {t.strip() for t in raw if t.strip()}

    VIBES = {
        "romantic": "romantic",
        "relaxed": "relaxed",
        "avoid crowds": "quiet",
        "avoid crowd": "quiet",
        "no crowds": "quiet",
        "quiet": "quiet",
        "fine dining": "fine dining",
        "finedining": "fine dining",
        "casual dining": "casual",
        "casual": "casual",
    }

    # simple phrase matching (handles "avoid crowds" and "fine dining" in free text)
    text = preferences.lower()

    def has(term: str) -> bool:
        return (term in text) if (" " in term) else (term in tokens)

    out: List[str] = []
    for key, normalized in VIBES.items():
        if (" " in key and key in text) or (key in tokens):
            if normalized not in out:
                out.append(normalized)
        if len(out) >= max_terms:
            break

    return out

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

def merge_unique_by_address(primary, secondary, limit):
    def key(x):
        return (x.get("place_id") or x.get("address") or "").lower().strip()
    
    seen = set(key(p) for p in primary if key(p))
    out = list(primary)
    for x in secondary:
        k = key(x)
        if not k or k in seen:
            out.append(x)
            seen.add(k)
        if len(out) >= limit:
            break
    return out

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
    - Fills act.map_url deterministically
    - Computes travel_mode / distance_from_prev / duration_minutes
      - Uses per-day NxN matrix when day unique locations <= cap
      - Falls back to pairwise per-leg when day exceeds cap or matrix missing values
    - Sets itinerary.total_distance_km
    """
    total_km = 0.0

    # Fill map_url first (deterministic)
    for day in itinerary.days:
        for act in day.activities:
            meta = addr_to_meta.get(act.location)
            act.map_url = maps_search_url(act.location, meta.get("place_id") if meta else None)

    # Per day routing matrix
    for day in itinerary.days:
        # collect unique locations for this day (in order)
        day_locs: List[str] = []
        seen: set[str] = set()

        for act in day.activities:
            loc = (act.location or "").strip()
            if loc and loc not in seen:
                seen.add(loc)
                day_locs.append(loc)

        use_matrix = prefer_single_matrix and (len(day_locs) <= nxn_destination_cap)

        routes: Dict[str, Any] = {}
        if use_matrix:
            routes = route_tool.run(
                origin=origin,
                destinations=day_locs,
                modes=allowed_modes,
                max_results=len(day_locs),
                return_matrix=True,
            )

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

        activity_terms = extract_activity_terms(preferences or "")

        # --------------------
        # 1) Python: Maps (once)
        # --------------------
        # cuisine_preferences is optional and your maps tool expects a dict keyed by meals,
        # but we do not "guess cuisines" from preferences; we keep it None for neutrality.
        cuisine_preferences = build_meal_hints(preferences or "")
        
        places_raw = maps_tool.run(
            location=location,
            activities=activity_terms if (preferences and activity_terms) else None,
            cuisine_preferences=cuisine_preferences,
            max_results_per_query=20,
        )

        #Fallback: if any of the meals are missing, requery meals only
        if not (places_raw.get("breakfast") and places_raw.get("lunch") and places_raw.get("dinner")):
            fallback = maps_tool.run(
                location=location,
                activities=None,
                cuisine_preferences={"breakfast": [], "lunch": [], "dinner": []},
                max_results_per_query=30,
            )
            # merge in anything missing
            for k in ("breakfast", "lunch", "dinner"):
                if not places_raw.get(k):
                    places_raw[k] = fallback.get(k, [])

        print({k: len(v) for k, v in places_raw.items()})

        # Make places compact + deterministic
        per_cat = min(20, max(6, days_count)) #up to 20 per category
        k_activity = min(60, max(24, days_count * 8)) #rank more when trip is longer
        k_meal = min(30, max(6, days_count * 3)) # need >= days_count if no reuse

        places_compact = compact_places(places_raw, per_cat=per_cat)
        candidates = flatten_candidates(places_compact)
        addr_to_meta = place_lookup(places_compact) 

        meal_cands = [c for c in candidates if c.get("category") in ("breakfast", "lunch", "dinner")]
        act_cands = [c for c in candidates if c.get("category") not in ("breakfast", "lunch", "dinner")]

        bundle = {"breakfast": [], "lunch": [], "dinner": [], "activities": []}
        
        vibes = extract_vibes(preferences or "")

        diet_terms = ("vegan", "vegetarian", "gluten-free", "gluten free")
        pref_lower = (preferences or "").lower()
        diet_requested = any(t in pref_lower for t in diet_terms)

        for meal in ("breakfast", "lunch", "dinner"):
            per_meal = [c for c in meal_cands if c.get("category") == meal]

            vibe_str = ", ".join(vibes) if vibes else "any"
            diet_line = "Prioritize vegan/vegetarian/gluten-free suitability. " if diet_requested else ""
            meal_pref = (
                f"{preferences}. For {meal}: vibe={vibe_str}. "
                f"{diet_line}"
                "Prioritize local food and strong reviews."
            )

            ranked = semantic_rank(meal_pref, per_meal, top_k=k_meal)

            print(f"🍽️ {meal}: per_meal={len(per_meal)} ranked={len(ranked)} days={days_count}")

            # If ranking is too strict or insufficient returns, fill with high-quality defaults
            if len(ranked) < days_count:
                fallback_sorted = sorted(
                    per_meal,
                    key=lambda x: ((x.get("rating") or 0), (x.get("user_ratings_total") or 0)),
                    reverse=True,
                )
                ranked = merge_unique_by_address(ranked, fallback_sorted, limit=k_meal)

            bundle[meal] = ranked

        # -------------- Balanced activities ---------------
        prefs_l = (preferences or "").lower()

        must_cats: List[str] = []
        if "museum" in prefs_l:
            must_cats.append("museums")
        if "landmark" in prefs_l:
            must_cats.append("landmarks")
        if "park" in prefs_l:
            must_cats.append("parks")
        if "art" in prefs_l:
            must_cats.append("art")

        picked: List[Dict[str, Any]] = []
        picked_addr: set[str] = set()

        # enough unique options so the planner can actually pick them across days
        per_must = min(20, max(6, days_count * 3))  # e.g. 3–8 per must category

        for cat in must_cats:
           cat_cands = [c for c in act_cands if (c.get("category") or "").lower().strip() == cat]
           # rank within that category using a category-focused query
           ranked_cat = semantic_rank(f"{cat}. {preferences or ''}", cat_cands, top_k=per_must)
           for x in ranked_cat:
               addr = (x.get("address") or "").lower().strip()
               if not addr or addr in picked_addr:
                   continue
               picked.append(x)
               picked_addr.add(addr)
        
        # fill remaining with global ranking, but diversify so one category doesn't take over
        remaining_slots = max(0, k_activity - len(picked))
        ranked_all = semantic_rank(
            preferences or "",
            [c for c in act_cands if (c.get("address") or "").lower().strip() not in picked_addr],
            top_k=max(remaining_slots * 3, remaining_slots)  # overfetch for diversity
        )

        per_cat_cap = 3
        counts: Dict[str, int] = {}
        fill: List[Dict[str, Any]] = []

        for x in ranked_all:
            cat = (x.get("category") or "other").lower().strip()
            if counts.get(cat, 0) >= per_cat_cap:
                continue
            fill.append(x)
            counts[cat] = counts.get(cat, 0) + 1
            if len(fill) >= remaining_slots:
                break

        bundle["activities"] = picked + fill

        # debug
        print("✅ Activities bundle category counts:",
              {k: sum(1 for a in bundle["activities"] if (a.get("category") or "").lower() == k)
               for k in sorted({(a.get("category") or "").lower() for a in bundle["activities"]})})

        # --------------------
        # 2) Python: Weather (once)
        # --------------------
        print("➡️ Weather: calling weather_tool.run()")
        weather_raw = weather_tool.run(
            city=location,
            start_date=start_date_iso,
            end_date=end_date_iso,
        )
        print("✅ Weather: returned from weather_tool.run()")

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

        # ensure enough unique meal options for the number of days
        for m in ("breakfast", "lunch", "dinner"):
            if len(bundle.get(m, [])) < days_count:
                return (
                    "### Not enough meal options found\n"
                    f"Need at least {days_count} unique {m} venues, but only found {len(bundle.get(m, []))}. "
                    "Try broadening preferences or increasing max_results_per_query."
                )
        
        print("✅ Bundle sizes:", {m: len(bundle.get(m, [])) for m in ("breakfast","lunch","dinner")},
            "activities:", len(bundle.get("activities", [])))
            
        # --------------------
        # 5) LLM: Planner (no tools) — compact deterministic context
        # --------------------
        print("➡️ Planner: building context_json")
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
                "Each day MUST include exactly 1 breakfast, 1 lunch, and 1 dinner activity (choose from the provided meal lists).",
                "Each day should include 2–4 non-meal activities (in addition to breakfast/lunch/dinner), unless weather is severe or options are limited.",
                "If preferences mention museums, landmarks, parks, or art — and matching categories exist in places.activities — include at least 1 activity per day from those matched categories before choosing other activities.",
                "DO NOT reuse venues.",
                "Do not schedule bars before 12:00.",
                "Use weather to bias indoor vs outdoor choices (rain -> museums/indoor).",
                "Do not worry about distance/time fields; Python will compute travel metrics after planning.",
                "Still output travel_mode/distance_from_prev/duration_minutes fields (can be null).",
                "Prefer <= 9 unique locations PER DAY to keep routing efficient."
            ],
        }

        print("➡️ Planner: creating planning_task")
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

        print("➡️ Planner: kickoff starting")
        _ = planner_crew.kickoff()
        print("✅ Planner: kickoff finished")

        # Extract the Pydantic object (CrewAI stores in task.output)
        planned = getattr(planning_task, "output", None)
        planned_obj = getattr(planned, "pydantic", None) or getattr(planned, "output_pydantic", None)

        if planned_obj is None:
            # fallback: try parse raw json
            raw = getattr(planned, "raw", None) or str(planned)
            planned_obj = ItineraryModel.model_validate_json(raw)

        itinerary: ItineraryModel = planned_obj

        used = set()
        for day in itinerary.days:
            for act in day.activities:
                loc = (act.location or "").strip()
                if not loc:
                    continue
                if loc in used:
                    return "### Planner reused a venue\nTry increasing max_results_per_query or relax constraints."
                used.add(loc)

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
                "\n"
                "- For each day, render activities strictly in the JSON order.\n"
                "- IMPORTANT: Do NOT output travel_mode, distance_from_prev, or duration_minutes as separate labeled fields inside an activity block.\n"
                "- Travel info must appear ONLY as the single italic line below.\n"
                "\n"
                "- For each activity (index i) in a day:\n"
                "  - If i > 0: print this line as its own paragraph (NOT a bullet, no leading '-' or '###') immediately BEFORE the activity block:\n"
                "    *Travel from previous: {duration_minutes} min, {distance_from_prev} km ({travel_mode})*\n"
                "  - If i == 0: do NOT print any travel line, even if travel fields are present.\n"
                "\n"
                "- The travel line uses the CURRENT activity’s fields (travel_mode/distance_from_prev/duration_minutes) because they represent travel FROM the previous activity TO this activity.\n"
                "\n"
                "- Example (follow exactly):\n"
                "  *Travel from previous: 13 min, 0.96 km (walking)*\n"
                "  **10:00 - 12:00**  \n"
                "  **ArteBo - Contemporary Art gallery** (⭐ 4.9)  \n"
                "  Location: [Via S. Petronio Vecchio, 8/A, 40125 Bologna BO, Italy](https://...)\n"
                "\n"
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