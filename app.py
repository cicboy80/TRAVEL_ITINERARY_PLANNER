import os
import gradio as gr
import traceback
import json
import time
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
    reasoning=False,
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
    llm="gpt-4o-mini",
    temperature=0.3,
    verbose=False
)

# Core logic
def generate_itinerary(location, start_date, end_date, preferences, transport_modes):
    t0 = time.time()
    print("✅ Button clicked:", location, start_date, end_date, transport_modes, preferences)

    try:
        # -------------------- normalize dates --------------------
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
        modes_json = json.dumps(transport_modes)

        # -------------------- tasks (phase 1: retrieve + weather) --------------------
        retrieval_task = Task(
            description=f"Gather restaurants, landmarks, and activities for the trip in {location}.",
            expected_output="A JSON containing categorized lists of places.",
            agent=retriever_agent,
        )

        weather_task = Task(
            description=(
                f"Use the Weather Forecast Tool **once** to fetch a 7-day weather forecast covering the full period "
                f"from {start_date} to {end_date} for {location}. "
                "Do not call the tool multiple times. Instead, request all days in one forecast. "
                "Return a single JSON mapping each date → weather summary (max/min temperature, precipitation, and general condition)."
            ),
            expected_output="A JSON object mapping each date to temperature, precipitation, and condition summaries.",
            agent=weather_agent,
        )

        prefetch_crew = Crew(
            agents=[retriever_agent, weather_agent],
            tasks=[retrieval_task, weather_task],
            verbose=True,
        )

        _ = prefetch_crew.kickoff(inputs={
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
        })

        # -------------------- extract retrieval output --------------------
        def _get_task_raw(task: Task) -> Any:
            out = getattr(task, "output", None)
            if out is None:
                return None
            raw = getattr(out, "raw", out)
            # Some versions store a dict already; others store JSON string
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    return raw
            return raw

        retrieval_data = _get_task_raw(retrieval_task)
        weather_data = _get_task_raw(weather_task)

        if retrieval_data is None:
            raise RuntimeError("RetrieverTask produced no output; cannot build route destinations.")

        # -------------------- build top-10 destinations across categories --------------------
        # expected shape from your log:
        # {
        #   "restaurants": {"breakfast":[...], "lunch":[...], "dinner":[...]},
        #   "landmarks":[...], "museums":[...], "parks":[...]
        # }
        destinations: List[str] = []

        def _pick_addr(item: Any) -> Optional[str]:
            if not isinstance(item, dict):
                s = str(item).strip()
                return s if s else None
            addr = (item.get("formatted_address") or item.get("address") or "").strip()
            if addr:
                return addr
            name = (item.get("name") or "").strip()
            return f"{name}, {location}".strip(", ") if name else None

        # pull in a balanced sample: breakfast/lunch/dinner + landmarks/museums/parks
        restaurants = (retrieval_data.get("restaurants") or {}) if isinstance(retrieval_data, dict) else {}
        for meal in ("breakfast", "lunch", "dinner"):
            for x in (restaurants.get(meal) or [])[:2]:  # 2 each = 6
                addr = _pick_addr(x)
                if addr:
                    destinations.append(addr)

        for cat in ("landmarks", "museums", "parks"):
            for x in (retrieval_data.get(cat) or [])[:2]:  # 2 each = 6 (we'll trim to 10)
                addr = _pick_addr(x)
                if addr:
                    destinations.append(addr)

        # de-dup while preserving order
        seen = set()
        destinations = [d for d in destinations if not (d in seen or seen.add(d))]

        # cap to 10
        destinations = destinations[:10]

        if not destinations:
            raise RuntimeError("No usable destinations built from RetrieverTask output.")

        # -------------------- call Route Planner Tool ONCE in Python (guaranteed) --------------------
        print("🧭 Calling Route Planner Tool once (return_matrix=True)...")
        route_data_dict = route_tool.run(
            origin=location,
            destinations=destinations,
            modes=transport_modes,
            max_results=10,
            return_matrix=True,
        )

        # Make sure it's JSON-serializable string for injection into planner prompt
        route_data_json = json.dumps(route_data_dict, ensure_ascii=False)

        # -------------------- tasks (phase 2: plan + write) --------------------
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
                "from the places retrieved by the RetrieverTask. Use the user's preferences to compute semantic similarity and select top results.\n\n"

                "4️⃣ Use the route and distance data provided below (this is REAL tool output, not a suggestion):\n"
                f"ROUTE_DATA_JSON = {route_data_json}\n\n"
                "   IMPORTANT: ROUTE_DATA_JSON contains:\n"
                "   - stops: list[str] (indexable; includes origin at index 0)\n"
                "   - matrix: dict keyed by mode, each containing distance_km[i][j] and duration_min[i][j]\n\n"
                "   How to use it correctly:\n"
                "   A) stops = ROUTE_DATA_JSON['stops']\n"
                "   B) matrix = ROUTE_DATA_JSON['matrix']\n"
                "   C) Build idx once: idx = { stops[i]: i for i in range(len(stops)) }\n"
                "   D) Matching rule (critical): Each activity 'location' MUST be exactly one of the strings in stops (verbatim).\n"
                "   E) For each consecutive activity A -> B:\n"
                "      - i = idx[A.location], j = idx[B.location]\n"
                "      - choose mode:\n"
                "        • if walking allowed and matrix['walking'].distance_km[i][j] <= 2.0 -> walking\n"
                "        • else if public_transport allowed -> public_transport\n"
                "        • else if cycling allowed -> cycling\n"
                "        • else -> driving\n"
                "      - set:\n"
                "        • distance_from_prev = matrix[chosen_mode].distance_km[i][j]\n"
                "        • duration_minutes  = matrix[chosen_mode].duration_min[i][j]\n"
                "        • travel_mode       = chosen_mode\n\n"

                "5️⃣ **Adjust activity timestamps dynamically based on route durations.**\n"
                "   - Use matrix[travel_mode].duration_min[i][j] between consecutive selected stops.\n"
                "   - If travel between two events exceeds 20 minutes, delay the next event’s start time accordingly.\n"
                "   - Keep the day within 08:00–22:00; reschedule or drop the lowest-ranked activity if needed.\n"
                "6️⃣ Assign realistic timestamps (breakfast 08:00–09:00, lunch 13:00–14:00, dinner after 19:00) and adjust based on travel time.\n"
                "7️⃣ Ensure variety across days (don’t repeat the same activities).\n"
                "8️⃣ Include timestamps and travel metadata in the JSON output for each event.\n"
                "9️⃣ For each selected item include: name, category, start/end time, location, rating (if any), reasoning, weather_forecast, "
                "distance_from_prev, duration_minutes, travel_mode.\n\n"
                "🔟 Output MUST match ItineraryModel exactly (activities field, duration_minutes field, transport_modes list).\n"
                f"transport_modes must equal: {modes_json}\n"
            ),
            expected_output="A structured JSON itinerary with complete metadata and travel-aware timestamps for each activity.",
            context=[retrieval_task, weather_task],
            agent=planner_agent,
            output_pydantic=ItineraryModel,
            reasoning=False,
        )

        writing_task = Task(
            description=(
                "You are a professional travel writer. Given a structured itinerary JSON with detailed fields "
                "(rating, reasoning, distance_from_prev, weather_forecast), write an engaging Markdown itinerary.\n\n"
                f"Make sure the itinerary is covering the period {start_date} to {end_date}.\n\n"
                "At the top include:\n"
                f"**Trip Dates:** {start_date} → {end_date}\n\n"
                "Each day must include:\n"
                "- The date and weather summary from the JSON.\n"
                "- Chronological itinerary entries with start/end times.\n"
                "- Only describe places present in the JSON. No invention.\n"
                "- Include rating inline (⭐ 4.7) when present.\n"
                "- Include travel line when present (e.g. '*Travel to next: 22-minute walk, 1.1 km*').\n"
                "Return Markdown with headers per day and bold timestamps."
            ),
            expected_output="A richly detailed Markdown itinerary with times, ratings, and reasoning per event.",
            context=[planning_task],
            agent=writer_agent,
        )

        planning_crew = Crew(
            agents=[planner_agent, writer_agent],
            tasks=[planning_task, writing_task],
            verbose=True,
        )

        result = planning_crew.kickoff(inputs={
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "transport_modes": transport_modes,
            "transport_modes_str": transport_modes_str,
            "trip_duration_days": trip_duration_days,
        })

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

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        max_threads=1,
        ssr_mode=False,   # <- important: SSR is experimental
    )