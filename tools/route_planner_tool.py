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

    # Optional NxN matrix output
    return_matrix: bool = Field(
        default=False,
        description="If true, also return an NxN distance/duration matrix over [origin] + destinations.",
    )

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
                    or ""
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
            if s.startswith("[") and s.endswith("]"):
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
        "and public transport for longer trips if both modes are provided. "
        "Optionally returns an NxN matrix over [origin] + destinations."
    )

    args_schema = RoutePlannerToolSchema

    # -------------------- INTERNAL HELPERS --------------------

    def _distance_matrix(
        self,
        client: httpx.Client,
        origins: List[str],
        destinations: List[str],
        mode: str,
        api_key: str,
    ) -> Dict[str, Any]:
        """Call Google Distance Matrix for origins -> destinations in one request."""
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": "|".join(origins),
            "destinations": "|".join(destinations),
            "mode": mode,
            "key": api_key,
        }
        if mode == "transit":
            params["departure_time"] = int(time.time())

        # ---- cache key (transit depends on time bucket) ----
        origins_k = "|".join(o.strip().lower() for o in origins)
        dests_k = "|".join(d.strip().lower() for d in destinations)
        mode_k = mode.strip().lower()

        dep_bucket = ""
        if mode_k == "transit":
            dep_bucket = f"t{int(time.time() // 900)}"  # 15 min bucket

        cache_key = f"distmat::{mode_k}::{origins_k}::{dests_k}"
        if dep_bucket:
            cache_key = f"{cache_key}::{dep_bucket}"

        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        r = client.get(url, params=params, timeout=20.0)
        r.raise_for_status()
        data = r.json()

        status = data.get("status")
        if status != "OK":
            err = data.get("error_message", "")
            raise RuntimeError(f"DistanceMatrix status={status}: {err}")

        ttl = 30 * 60 if mode_k == "transit" else 24 * 3600
        cache.set(cache_key, data, ttl_seconds=ttl)
        return data

    def _parse_nxn(self, dm: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Distance Matrix response into NxN matrices (km/min), with None where not OK."""
        rows = dm.get("rows", [])
        dist_km: List[List[Optional[float]]] = []
        dur_min: List[List[Optional[float]]] = []

        for r in rows:
            elems = r.get("elements", [])
            drow: List[Optional[float]] = []
            trow: List[Optional[float]] = []

            for el in elems:
                if el.get("status") == "OK":
                    drow.append(round(el["distance"]["value"] / 1000, 2))
                    trow.append(round(el["duration"]["value"] / 60, 1))
                else:
                    drow.append(None)
                    trow.append(None)

            dist_km.append(drow)
            dur_min.append(trow)

        return {"distance_km": dist_km, "duration_min": dur_min}

    # -------------------- MAIN ENTRY --------------------

    def _run(
        self,
        origin: str,
        destinations: List[str],
        modes: Union[str, List[str]] = "walking",
        max_results: int = 10,
        return_matrix: bool = False,
    ) -> Union[List[Dict[str, Union[str, float]]], Dict[str, Any]]:

        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_MAPS_API_KEY in environment variables")

        # allow stringified lists
        if isinstance(modes, str) and modes.strip().startswith("["):
            try:
                modes = json.loads(modes.replace("'", '"'))
            except Exception:
                pass

        if isinstance(modes, str):
            modes_list = [modes.strip()]
        else:
            modes_list = [str(m).strip() for m in modes if m is not None]

        # normalize input labels -> Google API labels
        # user: public_transport/cycling -> api: transit/bicycling
        norm_modes: List[str] = []
        for m in modes_list:
            if m == "public_transport":
                norm_modes.append("transit")
            elif m == "cycling":
                norm_modes.append("bicycling")
            else:
                norm_modes.append(m)

        norm_modes = [m for m in norm_modes if m]  # sanitize

        # output labels expected by planner
        api_to_user = {
            "walking": "walking",
            "transit": "public_transport",
            "driving": "driving",
            "bicycling": "cycling",
        }

        # Destination list
        MAX_DM_DESTS = 25
        dests = [str(d).strip() for d in destinations if str(d).strip()][
            : min(max_results, MAX_DM_DESTS)
        ]

        if not dests:
            if not return_matrix:
                return []
            return {
                "origin": origin,
                "destinations": [],
                "stops": [origin],
                "modes_requested": [api_to_user[m] for m in norm_modes if m in api_to_user],
                "origin_routes": [],
                "matrix": {},
            }

        # NxN safety: origins*destinations should stay <= ~100 elements
        MAX_STOPS_FOR_NXN = 10  # origin + first 9 destinations

        with httpx.Client(timeout=20.0) as client:
            # 1) origin -> dests (1xN) per requested mode
            dm_by_mode: Dict[str, Dict[str, Any]] = {}
            for m in ("walking", "transit", "driving", "bicycling"):
                if m in norm_modes:
                    dm_by_mode[m] = self._distance_matrix(client, [origin], dests, m, api_key)

            results: List[Dict[str, Union[str, float]]] = []

            # Distance matrix rows[0].elements aligns with destination array
            for i, dest in enumerate(dests):
                best: Optional[Dict[str, Union[str, float]]] = None

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
                    best = {
                        "mode": "walking",
                        "distance_km": round(el_walk["distance"]["value"] / 1000, 2),
                        "duration_min": round(el_walk["duration"]["value"] / 60, 1),
                    }

                # Try public transport if walking route > 2km or not available
                el_transit = get_el("transit")
                if el_transit and (best is None or float(best["distance_km"]) > 2.0):
                    cand = {
                        "mode": "public_transport",
                        "distance_km": round(el_transit["distance"]["value"] / 1000, 2),
                        "duration_min": round(el_transit["duration"]["value"] / 60, 1),
                    }
                    if best is None or float(cand["duration_min"]) < float(best["duration_min"]):
                        best = cand

                # bicycling / driving if requested and nothing else worked
                if best is None:
                    for fallback_mode, label in (("bicycling", "cycling"), ("driving", "driving")):
                        el = get_el(fallback_mode)
                        if el:
                            best = {
                                "mode": label,
                                "distance_km": round(el["distance"]["value"] / 1000, 2),
                                "duration_min": round(el["duration"]["value"] / 60, 1),
                            }
                            break

                if best:
                    best["destination"] = dest
                    results.append(best)

            # 2) Optional: NxN matrix over [origin] + destinations (still within single tool run)
            if not return_matrix:
                return results

            stops = [origin] + dests
            if len(stops) > MAX_STOPS_FOR_NXN:
                stops = stops[:MAX_STOPS_FOR_NXN]

            nxn_by_mode: Dict[str, Any] = {}
            for m in ("walking", "transit", "driving", "bicycling"):
                if m in norm_modes:
                    dm_nxn = self._distance_matrix(client, stops, stops, m, api_key)
                    nxn_by_mode[api_to_user[m]] = self._parse_nxn(dm_nxn)

            return {
                "origin": origin,
                "destinations": dests,
                "stops": stops,  # indexable list used by planner
                "modes_requested": [api_to_user[m] for m in norm_modes if m in api_to_user],
                "origin_routes": results,  # origin -> each stop (best mode per stop)
                "matrix": nxn_by_mode,      # NxN per mode (planner keys!)
            }