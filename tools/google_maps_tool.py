import os
import httpx
from typing import List, Dict, Optional, Any, Union
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict, field_validator
from utils.cache import SQLiteCache

cache = SQLiteCache("/tmp/cache.sqlite")


class GoogleMapsToolSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location: str
    activities: Optional[Union[List[str], str]] = None
    cuisine_preferences: Optional[Union[Dict[str, str], List[str], str]] = None
    max_results_per_query: int = Field(default=20, ge=1, le=50)

    @field_validator("location", mode="before")
    @classmethod
    def _norm_location(cls, v: Any) -> str:
        if isinstance(v, dict):
            v = v.get("location") or v.get("value") or v.get("description") or ""
        return str(v).strip()

    @field_validator("activities", mode="before")
    @classmethod
    def _norm_activities(cls, v: Any):
        if v is None:
            return None
        if isinstance(v, dict):
            v = v.get("activities") or v.get("value") or v.get("description") or v
        if isinstance(v, str):
            # allow "art, craft beer, nightlife"
            items = [s.strip() for s in v.split(",")]
            return [s for s in items if s]
        if isinstance(v, list):
            out: List[str] = []
            for x in v:
                if x is None:
                    continue
                out.append(str(x).strip())
            return [s for s in out if s]
        return None

    @field_validator("cuisine_preferences", mode="before")
    @classmethod
    def _norm_cuisine(cls, v: Any):
        """
        Accepts:
          - {"breakfast":"pastry","lunch":"seafood",...}
          - "bolognese"
          - ["bolognese"]
        Normalizes to dict or None.
        """
        if v is None:
            return None
        if isinstance(v, dict):
            # normalize keys to lowercase
            return {str(k).strip().lower(): str(val).strip() for k, val in v.items() if val}
        if isinstance(v, list):
            # treat first item as a global cuisine hint
            s = str(v[0]).strip() if v else ""
            return {"breakfast": s, "lunch": s, "dinner": s} if s else None
        if isinstance(v, str):
            s = v.strip()
            return {"breakfast": s, "lunch": s, "dinner": s} if s else None
        return None


class GoogleMapsTool(BaseTool):
    """
    CrewAI compatible tool for querying Google Places Text Search API.
    Preference-driven: if you pass activities extracted from user preferences,
    it will search those terms too (without hardcoding bar/craft beer logic).
    """

    name: str = "Google Maps Places Tool"
    description: str = (
        "Searches for places of interest in a city using Google Places Text Search. "
        "Returns categorized lists (meals + base activities + optional preference activities)."
    )

    args_schema = GoogleMapsToolSchema

    def _run(
        self,
        location: str,
        activities: Optional[Union[List[str], str]] = None,
        cuisine_preferences: Optional[Union[Dict[str, str], List[str], str]] = None,
        max_results_per_query: int = 20,
    ) -> Dict[str, List[Dict]]:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_MAPS_API_KEY in environment variables")

        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

        meal_categories = ["breakfast", "lunch", "dinner"]
        base_activity_categories = ["museums", "parks", "landmarks"]

        # normalize inputs (in case tool is called directly without Pydantic)
        if isinstance(activities, str):
            activities = [s.strip() for s in activities.split(",") if s.strip()]
        if cuisine_preferences and not isinstance(cuisine_preferences, dict):
            # let schema handle normally; but keep a fallback
            s = str(cuisine_preferences).strip()
            cuisine_preferences = {"breakfast": s, "lunch": s, "dinner": s} if s else None

        extra_activities = activities or []

        # ✅ keep defaults AND add preference-driven extras
        categories: List[str] = []
        for c in (meal_categories + base_activity_categories + list(extra_activities)):
            c = str(c).strip()
            if c and c not in categories:
                categories.append(c)

        all_results: Dict[str, List[Dict]] = {}

        with httpx.Client(timeout=15.0) as client:
            for category in categories:
                cuisine = None

                if category in meal_categories:
                    cuisine = (cuisine_preferences or {}).get(category) if cuisine_preferences else None
                    query = (
                        f"{cuisine} {category} in {location}"
                        if cuisine
                        else f"{category} restaurants in {location}"
                    )
                else:
                    # preference-driven term, no hardcoded “bar/craft beer” expansions
                    query = f"{category} in {location}"

                # cache by query (safer than category-only)
                qk = query.strip().lower()
                cache_key = f"places::q::{qk}"
                cached = cache.get(cache_key)
                if cached is not None:
                    all_results[category] = cached
                    continue

                params = {"query": query, "key": api_key}
                resp = client.get(base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                places = [
                    {
                        "name": r.get("name"),
                        "category": category,
                        "rating": r.get("rating"),
                        "address": r.get("formatted_address"),
                        "lat": r.get("geometry", {}).get("location", {}).get("lat"),
                        "lng": r.get("geometry", {}).get("location", {}).get("lng"),
                        "user_ratings_total": r.get("user_ratings_total"),
                        "place_id": r.get("place_id"),
                    }
                    for r in data.get("results", [])[:max_results_per_query]
                ]

                all_results[category] = places
                cache.set(cache_key, places, ttl_seconds=7 * 24 * 3600)

        return all_results