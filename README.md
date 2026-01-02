---
title: Travel Itinerary Planner
emoji: 🐠
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered travel itinerary planner
---

# Travel Itinerary Planner 🧭✨

An interactive **Gradio** app that generates a day-by-day travel itinerary based on your destination, dates, preferences, and transport mode. It combines structured routing + POI retrieval with LLM-assisted planning to produce a clear, usable plan.

> ⚠️ Educational project only — use at your own discretion.

## What it does
- **Builds a multi-day itinerary** from start/end dates
- **Finds relevant places (POIs)** based on your preferences
- **Plans routes between stops** (e.g., walking/transit/driving)
- **Considers weather** to improve day planning (when enabled)
- **Ranks options semantically** to match your interests
- **Outputs a structured itinerary** you can copy or adapt

## Project structure
```text
.
├── app.py
├── models/
│   ├── __init__.py
│   └── itinerary_model.py
├── tools/
│   ├── __init__.py
│   ├── google_maps_tool.py
│   ├── route_planner_tool.py
│   ├── semantic_ranking_tool.py
│   └── weather_tool.py
├── utils/
│   ├── __init__.py
│   ├── cache.py
│   └── date_utils.py
├── requirements.txt
└── README.md
```

## Getting started (local)

### 1. Clone and install

```bash
git clone https://github.com/cicboy80/TRAVEL_ITINERARY_PLANNER.git
cd TRAVEL_ITINERARY_PLANNER
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Environmental variables

This project uses API keys for **OpenAI** and **GOOGLE_MAPS_API**.

Set required keys as environment variables, or create a local `.env` file

Required
- OPENAI_API_KEY
- GOOGLE_MAPS_API_KEY

Example:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_MAPS_API_KEY="..."
```
### 3. Run the app

```bash
python app.py
```

## Hugging Face Spaces: Secrets Setup

Add required keys under Settings → Secrets in your Space:

- OPENAI_API_KEY
- GOOGLE_MAPS_API_KEY

## Notes on security

- Never commit secrets (API keys, tokens, .env, certificates)
- Prefer platform secret managers (Hugging Face Secrets / cloud secret stores)

## Roadmap

- Add richer itinerary outputs (maps/export)
- Improve caching + rate-limit handling
- Add hotel/restaurant filters and budget constraints

## License

Apache-2.0