import os
import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from datetime import datetime, timedelta 

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Load environment variables
load_dotenv()

print("🔍 GOOGLE_MAPS_API_KEY found:", bool(os.getenv("GOOGLE_MAPS_API_KEY")))
print("🔍 OPENAI_API_KEY found:", bool(os.getenv("OPENAI_API_KEY")))

os.environ["LITELLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

# Confirm key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY")
if not os.getenv("GOOGLE_MAPS_API_KEY"):
    print("⚠️ Warning: Missing GOOGLE_MAPS_API_KEY")


from tools.google_maps_tool import GoogleMapsTool
from tools.route_planner_tool import RoutePlannerTool
from tools.weather_tool import WeatherTool
from tools.semantic_ranking_tool import SemanticRankingTool
from models.itinerary_model import ItineraryModel
from utils.date_utils import expand_dates

#Initialize Tools
def create_tools():
    """Instantiate tools only after environment is ready."""
    return {
        "maps": GoogleMapsTool(),
        "route": RoutePlannerTool(),
        "weather": WeatherTool(),
        "semantic": SemanticRankingTool(),
    }

tools = create_tools()
maps_tool = tools["maps"]
route_tool = tools["route"]
weather_tool = tools["weather"]
semantic_ranking_tool = tools["semantic"]


#Define agents

retriever_agent = Agent(
    role="Retriever agent",
    goal="Collect the best possible restaurants, landmarks, and attractions for the target location.",
    backstory="You are an expert travel agent that uses Google Maps to find top-rated places for trip planning.",
    tools=[maps_tool],
    llm="gpt-4o-mini",
    temperature=0.2,
    verbose=False
)

weather_agent = Agent(
    role="Weather Analyst Agent",
    goal="Fetch and interpret the weather forecast for {location} between {start_date} and {end_date} to make informed decisions about the activities",
    backstory="A weather expert that has experience deciding upon the activities for an itinerary based on the weather forecast",
    tools=[weather_tool],
    llm="gpt-4o-mini",
    temperature=0.2,
    verbose=False
)

route_agent = Agent(
    role="Route Planner Agent",
    goal="Calculate walking, driving, cycling, public transport times between trip locations to optimize daily routes.",
    backstory="A navigtion expert skilled at minimizing time and building efficient routes.",
    tools=[route_tool],
    llm="gpt-4o-mini",
    temperature=0.25,
    verbose=False
)

planner_agent = Agent(
    role = "Senior Travel Agent",
    goal="Design a balanced daily itinerary based on the user preferences and feedback from the retriever_agent, weather_agent, and route_agent."
         "You will also use semantic reasoning to match user preferences "
         "with the best places, balancing weather, distance, and experience quality.",
    backstory="You are an senior travel agent with a wealth of experience in intelligently designing travel plans "
              "that balance culture, food, and rest time depending upon the client's preferences.",
    tools=[semantic_ranking_tool],
    llm="gpt-5-mini",
    temperature=0.3,
    output_schema=ItineraryModel,
    reasoning=True,
    verbose=False
)

writer_agent = Agent(
    role="Itinerary Writer Agent",
    goal=(
        "Transform a structured JSON itinerary (produced by the Senior Travel Agent) "
        "into a vivid, well-written Markdown itinerary that accurately reflects the planned locations, "
        "times, and weather details without inventing new destinations."
    ),
    backstory=(
        "You are a professional travel writer with deep knowledge of destinations, "
        "adept at crafting engaging, realistic itineraries that reflect real cities, landmarks, and restaurants. "
        "You avoid speculation or fictional places unless explicitly provided."
    ),
    llm="gpt-4o",
    temperature=0.7,
    verbose=False
)

# Core logic
def generate_itinerary(location, start_date, end_date, preferences, transport_modes ):
    if isinstance(start_date, str):
        start_date = start_date.replace("/", "-")
        start_date = datetime.fromisoformat(start_date).date().isoformat()
    elif hasattr(start_date, "date"):
        start_date = start_date.date().isoformat()
    
    if isinstance(end_date, str):
        end_date = end_date.replace("/", "-")
        end_date = datetime.fromisoformat(end_date).date().isoformat()
    elif hasattr(end_date, "date"):
        end_date = end_date.date().isoformat()
    days, date_list = expand_dates(start_date, end_date)
    trip_duration_days = days
    
    if isinstance(transport_modes, str):
        transport_modes = [transport_modes]
    
    transport_modes_str = ", ".join(transport_modes)

    #Define the tasks

    retrieval_task = Task(
        description=f"Gather restaurants, landmarks, and activities for the trip in {location}.",
        expected_output="A JSON containing categorized lists of places.",
        agent=retriever_agent
    )

    weather_task = Task(
        description=(f"Use the Weather Forecast Tool **once** to fetch a 7-day weather forecast covering the full period from {start_date} to {end_date} for {location}."
        "Do not call the tool multiple times. Instead, request all days in one forecast."
        "Return a single JSON mapping each date → weather summary (max/min temperature, precipitation, and general condition)."),
        expected_output="A JSON object mapping each date to temperature, precipitation, and condition summaries.",
        agent=weather_agent,
    )

    route_task = Task(
        description=(
            f"From the RetrieverTask output, build a destinations list (strings) using the "
            f"top 10 places across categories (prefer formatted address if present, else name + city). "
            f"Use origin='{location}'. Then call Route Planner Tool exactly once with:\n"
            f"- origin: '{location}'\n"
            f"- destinations: <that list>\n"
            f"- modes: {transport_modes}\n"
            f"Return the tool output as JSON."
        ),
        expected_output="A JSON list of routes with mode, distance_km, duration_min, destination.",
        agent=route_agent,
        context=[retrieval_task]
    )

    planning_task = Task(
        description=(
            "You are the itinerary planning agent responsible for creating an optimized, personalized multi-day trip plan.\n\n"
            "Follow these reasoning steps carefully:\n"
            f"1️⃣ Plan a detailed itinerary for {location} covering all days from ({start_date} to {end_date}) "
            f"considering the user's preferences: {preferences}. Ensure you generate exactly {trip_duration_days} "
            f"daily itineraries, corresponding to the days between {start_date} and {end_date}.\n"
            "2️⃣ Review the weather forecast (from WeatherTask) to determine which days are best for outdoor vs indoor activities. "
            "If heavy rain or extreme temperatures are forecast, prioritize indoor attractions or museums.\n"
            "3️⃣ For each category (breakfast, lunch, dinner, activities), call the Semantic Ranking Tool to identify the best matches "
            "from the places retrieved by the RetrieverTask. Use the user's preferences to compute semantic similarity and select top results.\n"
            "4️⃣ Use the route and distance data (from RouteTask) to minimize total travel time between selected locations "
            "and determine approximate travel times between locations by considering the user's selected transport modes "
            "(walking, public_transport, or driving). Use walking for short distances (<2km) and public transport for longer ones.\n"
            "5️⃣ **Adjust activity timestamps dynamically based on route durations.**\n"
            "   - After computing routes, calculate cumulative travel time between activities.\n"
            "   - If travel between two events exceeds 20 minutes, delay the next event’s start time accordingly.\n"
            "   - Ensure total daily schedule remains within 08:00–22:00. If an activity exceeds this range, reschedule or drop the lowest-ranked one.\n"
            "6️⃣ Assign realistic timestamps to each event (e.g., breakfast 8:00–9:00, lunch 13:00–14:00, "
            "activities between meals, dinner after 19:00) and dynamically adjust based on travel time.\n"
            "7️⃣ Ensure variety across days (don’t repeat the same activities).\n"
            "8️⃣ Include timestamps and travel metadata in the JSON output for each event.\n"
            "9️⃣ For each selected item (breakfast, activity, lunch, dinner), include:\n"
            "   - name\n"
            "   - category\n"
            "   - start_time, end_time\n"
            "   - location (address)\n"
            "   - rating (if available)\n"
            "   - reasoning (why chosen)\n"
            "   - weather_forecast (if applicable)\n"
            "   - distance_from_prev (km)\n"
            "   - duration_minutes (if applicable)\n"
            "   - travel_mode (optional)\n\n"
            "🔟 Output format (MUST match ItineraryModel exactly):\n"
            "{\n"
            '  "destination": "<city/region>",\n'
            '  "trip_duration_days": <int>,\n'
            f'  "transport_modes": {transport_modes},\n'
            '  "start_date": "YYYY-MM-DD",\n'
            '  "end_date": "YYYY-MM-DD",\n'
            '  "traveler_profile": "<short preference summary>",\n'
            '  "days": [\n'
            "    {\n"
            '      "date": "YYYY-MM-DD",\n'
            '      "weather_summary": "<string>",\n'
            '      "summary": "<string>",\n'
            '      "activities": [\n'
            "        {\n"
            '          "name": "<place name>",\n'
            '          "category": "<breakfast|lunch|dinner|museum|park|landmark|...>",\n'
            '          "start_time": "HH:MM",\n'
            '          "end_time": "HH:MM",\n'
            '          "location": "<address>",\n'
            '          "map_url": "<optional>",\n'
            '          "rating": <optional float>,\n'
            '          "reasoning": "<optional>",\n'
            '          "weather_forecast": "<optional>",\n'
            '          "travel_mode": "<optional walking|public_transport|driving|cycling>",\n'
            '          "distance_from_prev": <optional float>,\n'
            '          "duration_minutes": <optional int>\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            '  "total_distance_km": <optional float>,\n'
            '  "notes": "<optional>"\n'
            "}\n"
            "⚠️ Use field name 'activities' (NOT events). Use 'duration_minutes' (NOT travel_duration_min). Use 'transport_modes' (NOT transport_mode)."
        ),
        expected_output="A structured JSON itinerary with complete metadata and travel-aware timestamps for each activity.",
        context=[retrieval_task, weather_task, route_task],
        agent=planner_agent,
        reasoning=True
    )



    writing_task = Task(
        description=(
            "You are a professional travel writer. Given a structured itinerary JSON with detailed fields "
            "(rating, reasoning, distance_from_prev, weather_forecast), write an engaging Markdown itinerary.\n\n"
            f"Make sure the itinerary is covering the period {start_date} to {end_date}.\n\n"
            "At the top of the Markdown output, include a summary line like:\n"
            f"'**Trip Dates:** {start_date} → {end_date} \n"
            "Each day must include:\n"
            "- The date and weather summary from the JSON.\n"
            "- Chronological itinerary entries with start and end times.\n"
            "Follow the timestamps, preserve logical sequencing, and describe only the actual restaurants, landmarks, "
            "and activities from the structured itinerary. "
            "For each activity, place or meal, include these details when present:\n"
            "- Rating: '⭐ 4.7' inline with restaurant or activity name\n"
            "- Distance/Travel time between items (e.g., '*Travel to next: 22-minute walk, 1.1 km*')\n"
            "- Reasoning when relevant ('Chosen for excellent reviews or unique view of the lake')\n\n"
            "Ensure each event lists time, name (with Google Maps link if available), address, and category."
            "Do not invent fictional places or events. "
            "If weather indicates rain, mention it contextually (e.g., 'Since the afternoon may rain, head indoors.').\n\n"
            "⚠️ Important: The itinerary must remain consistent with the city and locations in the JSON (e.g., Florence, Italy). "
            "Do not add generic beach or island content unless explicitly present in the JSON.\n\n"
            "Return your final output in **Markdown** format with headers for each day and bold timestamps."
        ),
        expected_output="A richly detailed Markdown itinerary with times, ratings, and reasoning per event.",
        context=[planning_task],
        agent=writer_agent
    )

    trip_crew = Crew(
        agents=[retriever_agent, weather_agent, route_agent, planner_agent, writer_agent],
        tasks=[retrieval_task, weather_task, route_task, planning_task, writing_task],
        verbose=False
    )

    result = trip_crew.kickoff(inputs = {
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "transport_modes": transport_modes_str,
        "trip_duration_days": trip_duration_days
    })

    # Safely extract the result
    markdown_itinerary = (
        result if isinstance(result, str)
        else getattr(result, "raw", None) or str(result)
    )
    return markdown_itinerary

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧭 AI-Powered Travel Itinerary Planner")
    gr.Markdown("Plan optimized, weather-aware trips powered by multi-agent reasoning 🌍")

    with gr.Row():
        location = gr.Textbox(label="Destination", placeholder="e.g. Venice, Italy")
        transport_modes = gr.CheckboxGroup(
            ["walking", "public_transport", "driving", "cycling"],
            label = "transport_modes",
            value = ["walking"],
            info="Select one or multiple modes for route optimization"
        )

    with gr.Row():
        start_date = gr.DateTime(label="Start Date", include_time=False, type="datetime")
        end_date = gr.DateTime(label="End Date", include_time=False, type="datetime")
    
    preferences = gr.Textbox(label="Your Preferences", placeholder="art, local food, relaed pace, avoid queues")

    generate_btn = gr.Button("📝 Generate Itinerary")
    itinerary_markdown=gr.Markdown(label="🔖Your Personalized Itinerary")

    generate_btn.click(
        fn=generate_itinerary,
        inputs=[location, start_date, end_date, preferences, transport_modes],
        outputs=[itinerary_markdown]
    )

if __name__ == "__main__":
    demo.launch()
