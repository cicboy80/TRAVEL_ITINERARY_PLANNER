import os
import httpx
from typing import List, Dict, Optional
from crewai_tools import RagTool

class GoogleMapsTool(RagTool):
    """
    CrewAI compatible toool for querying Google Places Api
    """

    name: str = "Google Maps Places Tool"
    description: str = (
        "Searches for nearby places of interest (restaurants, lanmarks, museums etc.)"
        "in a specified city using the Google Places API."
    )

    def _run(
            self,
            location: str,
            activities: Optional[List[str]] = None,
            cuisine_preferences: Optional[Dict[str, str]] = None,
            max_results_per_query: int = 20
    ) -> Dict[str, List[Dict]]:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_MAPS_API_KEY in environment variables")
        
        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        categories = ["breakfast", "lunch", "dinner"] + (activities or ["museums", "parks", "landmarks"])
        all_results = {}
        with httpx.Client(timeout=15.0) as client:
            for category in categories:
                if category in ["breakfast", "lunch", "dinner"]:
                    cuisine = cuisine_preferences.get(category) if cuisine_preferences else None
                    query = f"{cuisine} {category} in {location}" if cuisine else f"{category} restaurants in {location}"
                else:
                    query = f"{category} in {location}"

                params = {"query": query, "key": api_key}
                resp = client.get(base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                all_results[category] = [
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
               
            return all_results