import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Type
from datetime import date, timedelta


# 🧾 Input Schema
class WeatherToolInput(BaseModel):
    """Schema defining accepted inputs for the Weather Forecast Tool."""
    city: Optional[str] = Field(None, description="City name, e.g. 'Florence, Italy'.")
    latitude: Optional[float] = Field(None, description="Latitude in decimal degrees.")
    longitude: Optional[float] = Field(None, description="Longitude in decimal degrees.")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")

# 🌤️ CrewAI Tool
class WeatherTool(BaseTool):
    """Fetches multi-day weather forecasts using the Open-Meteo API."""

    name: str = "Weather Forecast Tool"
    description: str = (
        "Fetches weather forecasts for a given city or coordinates using Open-Meteo. "
        "Includes temperature highs/lows and precipitation per day."
    )

    args_schema: Type[BaseModel] = WeatherToolInput

    def _run(
        self,
        city: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch daily weather data between start_date and end_date for the given city."""

        if not city and (latitude is None or longitude is None):
            raise ValueError("Either 'city' or coordinates (latitude, longitude) must be provided.")

        # Compute forecast duration
        if start_date and end_date:
            sd = date.fromisoformat(start_date)
            ed = date.fromisoformat(end_date)
            forecast_days = (ed - sd).days + 1
        else:
            forecast_days = 7

        # Geocode if needed
        if city and (latitude is None or longitude is None):
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
            geo_data = httpx.get(geo_url, timeout=10.0).json()
            if not geo_data.get("results"):
                raise ValueError(f"Could not geocode city: {city}")
            top = geo_data["results"][0]
            latitude, longitude = top["latitude"], top["longitude"]
            city = top.get("name", city)

        # Fetch weather
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
        }

        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date

        resp = httpx.get(base_url, params=params, timeout=10.0)
        data = resp.json()
        days = data.get("daily", {})

        forecasts = []
        n_days = min(len(days.get("time", [])), forecast_days)
        for i in range(n_days):
            forecasts.append({
                "date": days["time"][i],
                "temp_max": days["temperature_2m_max"][i],
                "temp_min": days["temperature_2m_min"][i],
                "precipitation_mm": days.get("precipitation_sum", [0]*n_days)[i],
            })

        return {
            "city": city,
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "forecast_days": forecast_days,
            "forecasts": forecasts
        }