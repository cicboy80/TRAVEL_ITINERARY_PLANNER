import os
import httpx
from typing import List, Dict, Union, Optional, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, ConfigDict
import time
import random
import json
from utils.cache import SQLiteCache

cache = SQLiteCache("/tmp/cache.sqlite")

class RoutePlannerToolSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    origin: str
    destinations: List[str]
    modes: Union[str, List[str]] = Field(default="walking")
    max_results: int = Field(default=10, ge=1, le=50)

    @field_validator("origin", mode="before")
    @classmethod
    def normalize_origin(cls, v: Any):
        if isinstance(v, dict):
            v = v.get("origin") or v.get("value") or v.get("description") or ""
        return str(v).strip()

    @field_validator("destinations", mode="before")
    @classmethod
    def normalize_destinations(cls, v: Any):
        # unwrap common CrewAI/LLM shapes
        if isinstance(v, dict):
            if "destinations" in v:
                v = v["destinations"]
            elif "description" in v:
                v = v["description"]

        # Accept messy inputs from the LLM and coerce to list[str]
        if isinstance(v, str):
            s = v.strip()

            if len(s) > 300 and not (s.startswith("[") and s.endswith("]")):
                return []
            
            if s.startswith("[") and s.endswith("]"):
                try:
                    s_json = s.replace("'", '"')
                    parsed = json.loads(s_json)
                    if isinstance(parsed, list):
                        v = parsed
                    else:
                        v = [s]
                except Exception:
                    v = [s]
            else:
                v = [s]

        # If it's a dict, try common address/name fields
        elif isinstance(v, dict):
            s = (
                v.get("formatted_address")
                or v.get("address")
                or v.get("name")
                or ""
            )
            s = str(s).strip()
            v = [s] if s else []

        # if already a list, flatten nested lists
        if isinstance(v, list):
            flat: List[Any] = []
            for x in v:
                if isinstance(x, list):
                    flat.extend(x)
                else:
                    flat.append(x)
            v = flat
        
        # If already a list sanitize items
        if not isinstance(v, list):
            return v
        
        out: List[str] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, dict):
                s = (
                    item.get("formatted_address")
                    or item.get("address")
                    or item.get("name")
                    or""
                )
            else:
                s = item
                
            s = str(s).strip()
            if not s or s.isdigit():
                continue
            out.append(s)

        return out

    @field_validator("modes", mode="before")
    @classmethod
    def normalize_modes(cls, v: Any):
        # unwrap common CrewAI/LLM shapes
        if isinstance(v, dict):
            v = v.get("modes") or v.get("value") or v.get("description") or v

        if v is None:
            return "walking"
        
        if isinstance(v, str):
            s = v.strip()
            # handle stringified list like '["walking","public_transport"]' or "['walking','public_transport']"
            if (s.startswith("[") and s.endswith("]")):
                try:
                    s_json = s.replace("'", '"')
                    parsed = json.loads(s_json)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if x is not None]
                except Exception:
                    pass
            return s

        if isinstance(v, list):
            return [str(x).strip() for x in v if x is not None]
        
        return "walking"

class RoutePlannerTool(BaseTool):
    """
    CrewAI-compatible tool for computing optimal routes between locations
    using Google Directions API with multi-mode transport support.
    """

    name: str = "Route Planner Tool"
    description: str = (
        "Computes walking, public transport, cycling, or driving routes between locations. "
        "Automatically selects walking for short distances (<2km) and public transport for longer trips "
        "if both modes are provided."
    )

    args_schema = RoutePlannerToolSchema

    def _run(
        self,
        origin: str,
        destinations: List[str],
        modes: Union[str, List[str]] = "walking",
        max_results: int = 10
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Calculates travel routes between a starting location and multiple destinations.

        Args:
            origin (str): Starting location name or address.
            destinations (List[str]): List of destinations to route to.
            modes (str or List[str]): One or multiple transport modes ('walking', 'driving', 'public_transport').
            max_results (int): Max number of routes to compute.

        Returns:
            List[Dict]: A list of dictionaries with mode, distance_km, duration_min, and destination.
        """

        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_MAPS_API_KEY in environment variables")

        if isinstance(modes, str) and modes.strip().startswith("["):
            try:
                modes = json.loads(modes.replace("'", '"'))
            except Exception:
                pass

        if isinstance(modes, list):
            modes = [str(x).strip() for x in modes if x is not None]
            modes = [m for m in modes if m]
            modes = ["transit" if m == "public_transport" else m for m in modes]

        if isinstance(modes,str):
            modes = [modes]

        if not destinations:
            return []

        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        results = []

        with httpx.Client(timeout=20.0) as client:
            for dest in destinations[:max_results]:
                dest = str(dest).strip()
                if not dest:
                    continue
                best_route = None

                # Try walking first
                if "walking" in modes:
                    walk_params = {"origin": origin, "destination": dest, "mode": "walking", "key": api_key}
                    walk_data = self._fetch_route(client, base_url, walk_params)
                    if walk_data:
                        best_route = {**walk_data, "mode": "walking"}

                # Try public transport if walking route > 2km or not available
                if ("public_transport" in modes or "transit" in modes) and (
                    not best_route or best_route["distance_km"] > 2
                ):
                    transit_params = {
                        "origin": origin,
                        "destination": dest,
                        "mode": "transit",
                        "transit_mode": "bus|subway|train|tram",
                        "key": api_key
                    }

                    transit_params["departure_time"] = "now"

                    transit_data = self._fetch_route(client, base_url, transit_params)
                    if transit_data:
                        if not best_route or transit_data["duration_min"] < best_route["duration_min"]:
                            best_route = {**transit_data, "mode": "public_transport"}
            
                # Cycling if included
                if "cycling" in modes and not best_route:
                    bike_params = {"origin": origin, "destination": dest, "mode": "bicycling", "key": api_key}
                    bike_data = self._fetch_route(client, base_url, bike_params)
                    if bike_data:
                        best_route = {**bike_data, "mode": "cycling"}

                # Try driving if included and others unavailable
                if "driving" in modes and not best_route:
                    drive_params = {"origin": origin, "destination": dest, "mode": "driving", "key": api_key}
                    drive_data = self._fetch_route(client, base_url, drive_params)
                    if drive_data:
                        best_route = {**drive_data, "mode": "driving"}

                if best_route:
                    best_route["destination"] = dest
                    results.append(best_route)

            return results

    def _fetch_route(self, client: httpx.Client, base_url: str, params: Dict) -> Optional[Dict]:
        """Fetch and parse route data from Google Directions API."""
        origin = str(params.get("origin", "")).strip().lower()
        dest = str(params.get("destination", "")).strip().lower()
        mode = str(params.get("mode", "")).strip().lower()
        transit_mode = str(params.get("transit_mode", "")).strip().lower()

        # Transit results depend on departure_time, so include a time bucket in the cache key
        departure_time = params.get("departure_time")
        dep_bucket = ""
        if mode == "transit":
            if departure_time == "now":
                dep_bucket = f"t{int(time.time() // 900)}" #15-min bucket
            elif departure_time is not None:
                dep_bucket = f"t{departure_time}"

        key = f"route::{origin}::{dest}::{mode}::{transit_mode}"
        if dep_bucket:
            key = f"{key}::{dep_bucket}"
        
        cached = cache.get(key)
        if cached is not None:
            if isinstance(cached, dict) and cached.get("__no_route__"):
                return None
            return cached
        
        # Cache TTL: transit should be short-lived, other are longer
        success_ttl = 30 * 60 if mode == "transit" else 24 * 3600
        negative_ttl = 10 * 60 # avoid hammering bad pairs
        
        for attempt in range(3):
            try:
                response = client.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                status = data.get("status", "")

                # transient: retry, don't negative-cache
                if status in ("OVER_QUERY_LIMIT", "UNKNOWN_ERROR"):
                    raise RuntimeError(f"Transient Directions status: {status}")
                
                # hard auth/config issue: fail loudly
                if status == "REQUEST_DENIED":
                    raise RuntimeError(f"Directions REQUEST_DENIED: {data.get('error_message', '')}".strip())
                

                if data.get("status") != "OK" or not data.get("routes"):
                    #cache negative results briefly to avoid hammering
                    cache.set(key, {"__no_route__": True}, ttl_seconds=negative_ttl)
                    return None

                leg = data["routes"][0]["legs"][0]
                out = {
                    "distance_km": round(leg["distance"]["value"] / 1000, 2),
                    "duration_min": round(leg["duration"]["value"] / 60, 1),
                    "start_address": leg["start_address"],
                    "end_address": leg["end_address"]
                }

                cache.set(key, out, ttl_seconds=success_ttl)
                return out
            
            except (httpx.TimeoutException, httpx.RequestError) as e:
                print(f"Directions network/timeout (attempt {attempt+1}/3): {e}")

            except httpx.HTTPStatusError as e:
                # retry only for transient HTTP codes
                code = e.response.status_code
                if code in (429, 500, 502, 503, 504):
                    print(f"Directions HTTP {code} (attempt {attempt+1}/3): retrying")
                else:
                    print(f"Directions HTTP {code}: {e}")
                    return None
            
            except RuntimeError:
                # REQUEST_DENIED / hard failures should surface immediately
                raise

            except Exception as e:
                print(f"Error fetching route (attempt {attempt+1}/3): {e}")
                
            # expinential-ish backoff + jitter
            time.sleep((0.6 * (attempt +1)) + random.random() * 0.3)

        return None