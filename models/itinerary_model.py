from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta 


# 🗺️ Core activity element
class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity or restaurant")
    category: str = Field(..., description="Type of place, e.g. restaurant, museum, park")
    start_time: str = Field(..., description="Planned start time in HH:MM format")
    end_time: str = Field(..., description="Planned end time in HH:MM format")
    location: str = Field(..., description="Address or landmark location")
    map_url: Optional[str] = Field(None, description="Google Maps URL for the location")
    rating: Optional[float] = Field(None, description="Average rating score (if available)")
    reasoning: Optional[str] = Field(None, description="Reason why this activity was selected")
    weather_forecast: Optional[str] = Field(None, description="Expected weather at this time/location")
    travel_mode: Optional[str] = Field(None, description="Mode used to reach this activity (walking/public_transport/driving/cycling)")
    distance_from_prev: Optional[float] = Field(None, description="Distance from previous activity in kilometers")
    duration_minutes: Optional[int] = Field(None, description="Automatically computed duration in minutes.")

# 📅 Daily itinerary
class DayPlan(BaseModel):
    date: str = Field(..., description="Date of the plan, ISO format YYYY-MM-DD")
    weather_summary: Optional[str] = Field(None, description="Daily summary: temperature, precipitation, etc.")
    summary: Optional[str] = Field(None, description="High-level overview of the day")
    activities: List[Activity] = Field(..., description="Ordered list of daily activities and meals")

# 🌍 Entire trip plan
class ItineraryModel(BaseModel):
    destination: str = Field(..., description="City or region of the trip")
    trip_duration_days: int = Field(..., description="Number of days in the itinerary")
    transport_modes: List[str] = Field(default_factory=list, description="Allowed modes: walking, public_transport, driving, cycling")
    start_date: Optional[str] = Field(None, description="Trip start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD).")
    traveler_profile: Optional[str] = Field(None, description="Short description of traveler preferences.")
    days: List[DayPlan] = Field(..., description="List of daily plans in chronological order.")
    total_distance_km: Optional[float] = Field(None, description="Total travel distance across the itinerary")
    notes: Optional[str] = Field(None, description="Additional recommendations or general notes")
