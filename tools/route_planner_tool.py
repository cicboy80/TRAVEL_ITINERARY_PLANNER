import os
import httpx
from typing import List, Dict, Union, Optional
from crewai.tools import BaseTool


class RoutePlannerTool(BaseTool):
    """
    CrewAI-compatible tool for computing optimal routes between locations
    using Google Directions API with multi-mode transport support.
    """

    name: str = "Route Planner Tool"
    description: str = (
        "Computes walking, public transport, or driving routes between locations. "
        "Automatically selects walking for short distances (<2km) and public transport for longer trips "
        "if both modes are provided."
    )

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

        if isinstance(modes, str):
            modes = [modes]

        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        results = []

        with httpx.Client(timeout=20.0) as client:
            for dest in destinations[:max_results]:
                best_route = None

                # Try walking first
                if "walking" in modes:
                    walk_params = {"origin": origin, "destination": dest, "mode": "walking", "key": api_key}
                    walk_data = self._fetch_route(base_url, walk_params)
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
                    transit_data = self._fetch_route(base_url, transit_params)
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
                    drive_data = self._fetch_route(base_url, drive_params)
                    if drive_data:
                        best_route = {**drive_data, "mode": "driving"}

                if best_route:
                    best_route["destination"] = dest
                    results.append(best_route)

            return results

    def _fetch_route(self, client: httpx.Client, base_url: str, params: Dict) -> Optional[Dict]:
        """Fetch and parse route data from Google Directions API."""
        try:
            response = client.get(base_url, params=params, timeout=20.0)
            data = response.json()

            if data["status"] != "OK" or not data.get("routes"):
                return None

            leg = data["routes"][0]["legs"][0]
            distance_km = leg["distance"]["value"] / 1000
            duration_min = leg["duration"]["value"] / 60

            return {
                "distance_km": round(distance_km, 2),
                "duration_min": round(duration_min, 1),
                "start_address": leg["start_address"],
                "end_address": leg["end_address"]
            }

        except Exception as e:
            print(f"Error fetching route: {e}")
            return None
