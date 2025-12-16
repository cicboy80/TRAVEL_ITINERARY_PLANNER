import os
import httpx
from typing import List, Dict, Union, Optional, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, ConfigDict
import time
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
    using Google Distance Matrix with multi-mode transport support.
    """

    name: str = "Route Planner Tool"
    description: str = (
        "Computes walking, public transport, cycling, or driving routes between locations. "
        "Uses Google Distance Matrix for speed. Automatically selects walking for short distances (<2km) "
        "and public transport for longer trips if both modes are provided."
    )

    args_schema = RoutePlannerToolSchema

    def _distance_matrix(
            self,
            client: httpx.Client,
            origin: str,
            destinations: List[str],
            mode: str,
            api_key: str
    ) -> Dict[str, Any]:
        """Call Google Distance Matrix for origin -> multiple destinations in one request."""
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": origin,
            "destinations": "|".join(destinations),
            "mode": mode,
            "key": api_key,
        }
        if mode == "transit":
            params["departure_time"] = int(time.time())

        # cache key (transit depends on time bucket)
        origin_k = origin.strip().lower()
        dests_k = "|".join([d.strip().lower() for d in destinations])
        mode_k = mode.strip().lower()

        dep_bucket = ""
        if mode_k == "transit":
            dep_bucket = f"t{int(time.time() // 900)}" #15 min bucket

        cache_key = f"distmat::{origin_k}::{mode_k}::{dests_k}"
        if dep_bucket:
            cache_key = f"{cache_key}::{dep_bucket}"

        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        
        r = client.get(url, params=params, timeout=20.0)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "OK":
            status = data.get("status")
            err = data.get("error_message", "")
            raise RuntimeError(f"DistanceMatrix status={status}: {err}")

        # TTL: transit is short-lived; others are longer
        ttl = 30 * 60 if mode_k == "transit" else 24 * 3600
        cache.set(cache_key, data, ttl_seconds=ttl)
        return data  

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

        if isinstance(modes,str):
            modes_list = [modes.strip()]
        else:
            modes_list = [str(m).strip() for m in modes if m is not None]

        norm_modes: List[str] = []
        for m in modes_list:
            if m == "public_transport":
                norm_modes.append("transit")
            elif m == "cycling":
                norm_modes.append("bicycling")
            else:
                norm_modes.append(m)

        # sanitize
        norm_modes = [m for m in norm_modes if m]

        # Destination list
        MAX_DM_DESTS = 25
        dests = [str(d).strip() for d in destinations if str(d).strip()][:min(max_results, MAX_DM_DESTS)]
        if not dests:
            return []

        with httpx.Client(timeout=20.0) as client:
            dm_by_mode: Dict[str, Dict[str, Any]] = {}

            # call each requested mode once
            for m in ("walking", "transit", "driving", "bicycling"):
                if m in norm_modes:
                    dm_by_mode[m] = self._distance_matrix(client, origin, dests, m, api_key)

            results: List[Dict[str, Union[str, float]]] = []

            # Distance matrix rows[0].elements alighns with destination array
            for i, dest in enumerate(dests):
                best: Optional[Dict[str, Union[str, float]]] = None

                # Helper to extract one element
                def get_el(mode: str) -> Optional[Dict[str, Any]]:
                    dm = dm_by_mode.get(mode)
                    if not dm:
                        return None
                    try:
                        el = dm["rows"][0]["elements"][i]
                    except Exception:
                        return None
                    return el if el.get("status") == "OK" else None

                # Try walking first
                el_walk = get_el("walking")
                if el_walk:
                    walk_km = round(el_walk["distance"]["value"] / 1000, 2)
                    walk_min = round(el_walk["duration"]["value"] / 60, 1)
                    best = {
                        "mode": "walking",
                        "distance_km": walk_km,
                        "duration_min": walk_min,
                    }

                # Try public transport if walking route > 2km or not available
                el_transit = get_el("transit")
                if el_transit and (best is None or float(best["distance_km"]) > 2.0):
                    tr_km = round(el_transit["distance"]["value"] / 1000, 2)
                    tr_min = round(el_transit["duration"]["value"] / 60, 1)
                    cand = {
                        "mode": "public_transport",
                        "distance_km": tr_km,
                        "duration_min": tr_min,
                    }
                    if best is None or float(cand["duration_min"]) < float(best["duration_min"]):
                        best = cand

                # bicycling / driving if requested and nothing else worked
                if best is None:
                    for fallback_mode, label in (("bicycling", "cycling"), ("driving", "driving")):
                        el = get_el(fallback_mode)
                        if el:
                            km = round(el["distance"]["value"] / 1000, 2)
                            mn = round(el["duration"]["value"] / 60, 1)
                            best = {"mode": label, "distance_km": km, "duration_min": mn}
                            break

                if best:
                    best["destination"] = dest
                    results.append(best)              
                
            return results