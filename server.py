import os
import re
import json
import time
import uuid
import requests
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Load environment variables relative to the script directory
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path, override=True)

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate

try:
    from langchain_tavily import TavilySearch
    TAVILY_IMPORT_OK = True
except Exception:
    TAVILY_IMPORT_OK = False

# ----------------------------------------------------------------------------
# CONFIG & KEYS
# ----------------------------------------------------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

MAX_TOOL_HOPS = 6

RESTAURANT_KEYWORDS = [
    "restaurant", "restaurants", "cafe", "café", "coffee shop", "eatery",
    "eateries", "food", "eat", "dining", "brunch", "dinner spot",
    "date spot", "date spots", "places to eat", "bistro",
]

# ----------------------------------------------------------------------------
# AGENT STUFF (Copied and adapted from streamVersion2.py)
# ----------------------------------------------------------------------------
def get_llm():
    return ChatOpenAI(model="gpt-5-mini")

def get_tavily():
    if not TAVILY_IMPORT_OK or not TAVILY_KEY:
        return None
    return TavilySearch(max_results=4)

# ----------------------------------------------------------------------------
# TOOLS
# ----------------------------------------------------------------------------
# We modify tools slightly to return structured JSON so backend can capture updates.

@tool
def get_weather(city: str) -> str:
    """Get the current weather for the given city"""
    if not WEATHER_KEY:
        return json.dumps({
            "error": "Weather tool unavailable: OPENWEATHER_API_KEY is not set.",
            "dossier_updates": {}
        })
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": WEATHER_KEY, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if response.status_code != 200:
            return json.dumps({
                "error": f"Error {response.status_code}: {data}",
                "dossier_updates": {}
            })
        
        weather_info = (
            f"The current weather in {data['name']} is "
            f"{data['weather'][0]['description']} with a temperature of "
            f"{data['main']['temp']}°C, feels like {data['main']['feels_like']}°C, "
            f"humidity at {data['main']['humidity']}%, and wind speed "
            f"{data['wind']['speed']} m/s."
        )
        
        return json.dumps({
            "info": weather_info,
            "dossier_updates": {
                "city": data.get("name", city),
                "weather": {
                    "temp": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "desc": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind": data["wind"]["speed"]
                }
            }
        })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to get weather: {e}",
            "dossier_updates": {}
        })

@tool
def get_news(city: str) -> str:
    """Get latest important city updates for planning a date like traffic, road closures, crime, events, and popular restaurants."""
    tavily = get_tavily()
    if tavily is None:
        return "News tool unavailable: TAVILY_API_KEY is not set or langchain-tavily is not installed."
    query = (
        f"Latest news in {city} about road closures, traffic, crimes, "
        f"events, restaurants, nightlife, and public safety"
    )
    try:
        results = tavily.invoke(query)
        if not results:
            return f"No recent news found for {city}."
        summaries = []
        for result in results.get("results", []):
            title = result.get("title", "No title")
            content = result.get("content", "No details")
            summaries.append(f"{title}: {content}")
        return "\n-----------------------\n".join(summaries) if summaries else f"No recent news found for {city}."
    except Exception as e:
        return f"Failed to fetch news: {e}"

def get_coordinates(city: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "my-date-planner-agent"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        if not data:
            return None
        return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
    except Exception:
        return None

def get_ip_city():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=6)
        if r.status_code == 200:
            data = r.json()
            city = data.get("city")
            if city:
                return city
    except Exception:
        pass
    try:
        r = requests.get("http://ip-api.com/json/", timeout=6)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success" and data.get("city"):
                return data.get("city")
    except Exception:
        pass
    return None

def mentions_restaurant_intent(text: str) -> bool:
    t = text.lower()
    return any(keyword in t for keyword in RESTAURANT_KEYWORDS)

def quick_extract_city(text: str):
    """Lightweight LLM check for whether the user explicitly named a city in this message."""
    llm = get_llm()
    prompt = (
        "Extract only the city name explicitly mentioned in the following message. "
        "If no city is mentioned, reply with exactly: NONE. "
        "Reply with the city name only, nothing else.\n\n"
        f"Message: {text}"
    )
    try:
        response = llm.invoke(prompt)
        value = response.content.strip().strip(".")
    except Exception:
        return None
    if not value or value.upper() == "NONE":
        return None
    return value

city_extract_prompt = PromptTemplate.from_template(
    "Extract the city name only from the following prompt: {prompt}"
)

@tool
def get_city(user_input: str) -> str:
    """Extract the city name from the user's prompt"""
    llm = get_llm()
    response = llm.invoke(city_extract_prompt.format(prompt=user_input))
    city = response.content.strip()
    return json.dumps({
        "info": f"Extracted city: {city}",
        "dossier_updates": {
            "city": city
        }
    })

@tool
def get_restaurants(city: str) -> str:
    """Get nearby restaurants in the given city using OpenStreetMap Overpass API"""
    coords = get_coordinates(city)
    if not coords:
        return f"Could not find coordinates for {city}"

    lat, lon = coords["lat"], coords["lon"]
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="restaurant"](around:3000,{lat},{lon});
      way["amenity"="restaurant"](around:3000,{lat},{lon});
      relation["amenity"="restaurant"](around:3000,{lat},{lon});
    );
    out center;
    """.strip()

    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    ]
    headers = {"User-Agent": "my-date-planner-agent"}

    for url in overpass_urls:
        try:
            response = requests.get(url, params={"data": query}, headers=headers, timeout=15)
            if response.status_code != 200:
                continue
            try:
                data = response.json()
            except Exception:
                continue

            restaurants, seen = [], set()
            for place in data.get("elements", []):
                name = place.get("tags", {}).get("name")
                if name and name not in seen:
                    seen.add(name)
                    restaurants.append(name)

            if restaurants:
                return "\n".join(restaurants[:10])
        except requests.exceptions.RequestException:
            continue

    return f"Restaurant lookup failed for {city}. Try again later."

TOOLS = {
    "get_weather": get_weather,
    "get_news": get_news,
    "get_city": get_city,
    "get_restaurants": get_restaurants,
}

SYSTEM_PROMPT = """
You are an intelligent AI date planning assistant.

Your job is to help users with date-related suggestions, plans, places, restaurants, weather awareness, and local city updates.

Behavior rules:
1. Understand the user's intent first.
2. Only call tools when needed.
3. Use get_city only if the city is unclear.
4. Use get_weather only if weather matters for the plan.
5. Use get_news only if safety, traffic, or events are relevant.
6. Use get_restaurants when the user asks for places, food, cafes, restaurants, or date spots.
7. Never force all tools unnecessarily.
8. If a tool fails, do not retry more than once.
9. If the user's message includes a note that their location/city was already detected, treat that as the city — do not ask for it again.

Response style:
- Keep responses short, natural, and conversational by default.
- Only give detailed structured plans when the user explicitly asks for a full plan, itinerary, complete date plan, or "plan my evening".
- For simple questions, answer in 2-5 lines.
- For recommendations, keep it concise and practical.
- Expand only when necessary.

Planning logic:
- Prefer indoor plans during bad weather.
- Avoid unsafe or high-traffic areas if news mentions them.
- Match recommendations to budget, vibe, and convenience.
- Prioritize comfort, romance, practicality, and realism.

Tone: Warm, smart, concise, helpful, and natural.
"""

FINAL_PROMPT = PromptTemplate.from_template("""
You are an intelligent AI date planning assistant.

Below is the full conversation history, including user preferences, city information, weather data, local news, and restaurant suggestions.

Conversation:
{messages}

Respond with STRICT JSON ONLY — no markdown, no code fences, no commentary before or after the JSON. The JSON must match exactly this schema:

{{
  "message": "<a short, warm, conversational reply to the user's latest message. 2-5 lines for simple questions, or a brief intro line when a full plan is also included.>",
  "plan": null OR {{
      "timeline": null OR [
          {{"time": "<e.g. 6:00 PM>", "activity": "<what happens at this time, one short sentence>"}}
      ] (only if the user explicitly asked for a full plan or itinerary; provide 3-6 ordered steps, each with a short clock time and a short activity description),
      "best_area": "<best area/location and why>",
      "best_time": "<best time and why>",
      "things_to_avoid": "<traffic, weather, safety, or budget pitfalls to avoid>",
      "backup_plan": "<an alternative option>"
  }},
  "restaurants": [
      {{"name": "<restaurant/cafe/place name>", "why": "<one short sentence on why this place fits>"}}
  ]
}}

Rules:
1. Set "plan" to null unless the user explicitly asked for a complete plan, itinerary, or full date plan. For simple questions, "plan" must be null.
2. Only include entries in "restaurants" when the user is asking about places, food, cafes, restaurants, or date spots, or when a restaurant naturally belongs in a full plan. Otherwise return an empty array [].
3. Prefer restaurant names that actually appeared in tool outputs in the conversation when available; do not contradict them.
4. Use weather only if it affects the plan. Use news only if safety, traffic, or events matter.
5. Prefer indoor plans during bad weather, outdoor only when weather is comfortable. Avoid unsafe/crowded areas mentioned in news. Match budget, vibe, convenience. Prioritize romance, comfort, practicality, realism.
6. Keep "message" natural and do not repeat tool outputs verbatim.
7. Output must be valid JSON: double quotes only, no trailing commas, no comments, no extra keys.
8. Each "timeline" step's "time" must be short (e.g. "6:00 PM"), and "activity" must be one short, concrete sentence — never a paragraph.

Tone inside the text fields: Warm, concise, romantic, practical, personalized.
""")

# ----------------------------------------------------------------------------
# SESSION STORE MANAGEMENT
# ----------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.chat_log = []  # User-facing chat logs
        self.pending_calls = []  # Tool calls waiting for approval
        self.dossier = {
            "city": None,
            "weather": None,
            "log": []  # Activity logs
        }
        self.hops = 0
        self.location_saved = False
        self.awaiting_location = False
        self.pending_user_text = ""

    def reset(self):
        self.__init__()

    def to_dict(self):
        return {
            "chat_log": self.chat_log,
            "dossier": self.dossier,
            "pending_calls": self.pending_calls,
            "location_saved": self.location_saved,
            "awaiting_location": self.awaiting_location,
            "pending_user_text": self.pending_user_text,
            "status": self.get_status()
        }

    def get_status(self):
        if self.awaiting_location:
            return "awaiting_location"
        if self.pending_calls:
            return "awaiting_tool_approval"
        return "idle"

SESSIONS: Dict[str, SessionState] = {}

def get_session(session_id: str) -> SessionState:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState()
    return SESSIONS[session_id]

# ----------------------------------------------------------------------------
# PARSING AGENT OUTPUT
# ----------------------------------------------------------------------------
def _extract_json_object(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None

def parse_final_json(text: str, session: SessionState):
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return None

    message = str(data.get("message") or "").strip()
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else None

    city = (session.dossier.get("city") or "").strip()
    restaurants = []
    for r in data.get("restaurants") or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        desc = str(r.get("why") or r.get("description") or "").strip()
        query = f"{name} {city}".strip()
        link = f"https://www.google.com/maps/search/{requests.utils.quote(query)}"
        restaurants.append({"name": name, "desc": desc, "link": link})

    return {"message": message, "plan": plan, "restaurants": restaurants}

# ----------------------------------------------------------------------------
# AGENT HOPPING ENGINE
# ----------------------------------------------------------------------------
def run_agent_step(session: SessionState):
    llm = get_llm()
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))
    
    response = llm_with_tools.invoke(session.messages)
    session.messages.append(response)

    if response.tool_calls and session.hops < MAX_TOOL_HOPS:
        session.pending_calls = []
        for call in response.tool_calls:
            session.pending_calls.append({
                "id": call["id"],
                "name": call["name"],
                "args": call["args"]
            })
    else:
        finalize_response(session)

def finalize_response(session: SessionState):
    llm = get_llm()
    final = llm.invoke(FINAL_PROMPT.format(messages=session.messages))
    raw = final.content
    structured = parse_final_json(raw, session)
    display_text = structured["message"] if structured and structured.get("message") else raw

    session.messages.append(AIMessage(content=raw))
    session.chat_log.append({
        "role": "assistant",
        "content": display_text,
        "structured": structured
    })
    session.pending_calls = []
    session.hops = 0

# ----------------------------------------------------------------------------
# FASTAPI BACKEND APP
# ----------------------------------------------------------------------------
app = FastAPI(title="Aisle Date Concierge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    action: str  # "message", "approve_tool", "deny_tool", "resolve_location", "reset"
    message: Optional[str] = None
    tool_call_id: Optional[str] = None
    city: Optional[str] = None

@app.get("/api/session")
def api_get_session():
    session_id = str(uuid.uuid4())
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "state": session.to_dict(),
        "system_status": {
            "openai": bool(OPENAI_KEY),
            "weather": bool(WEATHER_KEY),
            "tavily": bool(TAVILY_KEY) and TAVILY_IMPORT_OK
        }
    }

@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    session = get_session(payload.session_id)

    if payload.action == "reset":
        session.reset()
        return session.to_dict()

    try:
        if payload.action == "message":
            text = payload.message or ""
            if not text:
                raise HTTPException(status_code=400, detail="Message content empty.")

            session.chat_log.append({"role": "user", "content": text})

            needs_location = False
            if mentions_restaurant_intent(text):
                mentioned_city = quick_extract_city(text)
                if mentioned_city:
                    session.dossier["city"] = mentioned_city
                    session.location_saved = True
                elif not session.location_saved:
                    needs_location = True

            if needs_location:
                session.awaiting_location = True
                session.pending_user_text = text
                return session.to_dict()

            session.messages.append(HumanMessage(content=text))
            session.hops = 0
            run_agent_step(session)
            return session.to_dict()

        if payload.action == "resolve_location":
            city = payload.city or ""
            if not city:
                raise HTTPException(status_code=400, detail="City name is empty.")
            
            pending = session.pending_user_text or ""
            session.dossier["city"] = city
            session.location_saved = True
            session.awaiting_location = False
            session.pending_user_text = ""

            note_text = (
                f"{pending}\n\n(System note: the user's current city was detected as {city} "
                f"via device location. Use this as the city for this request — do not ask for "
                f"the city again unless the user names a different one later.)"
            )
            session.messages.append(HumanMessage(content=note_text))
            session.hops = 0
            run_agent_step(session)
            return session.to_dict()

        if payload.action in ["approve_tool", "deny_tool"]:
            if not session.pending_calls:
                raise HTTPException(status_code=400, detail="No pending tools to approve/deny.")
            
            call = session.pending_calls[0]
            ts = time.strftime("%H:%M:%S")
            tool_name = call["name"]

            if payload.action == "deny_tool":
                content = f"Tool call '{tool_name}' was denied by the user. Proceed without this information."
                session.dossier["log"].append({
                    "name": tool_name,
                    "status": "denied",
                    "time": ts
                })
            else:
                try:
                    args = call["args"]
                    tool_func = TOOLS[tool_name]
                    result = tool_func.invoke(args)
                    
                    try:
                        result_data = json.loads(str(result))
                        if isinstance(result_data, dict) and "dossier_updates" in result_data:
                            content = result_data.get("info", str(result))
                            updates = result_data["dossier_updates"]
                            for key, val in updates.items():
                                if isinstance(val, dict) and isinstance(session.dossier.get(key), dict):
                                    session.dossier[key].update(val)
                                else:
                                    session.dossier[key] = val
                        else:
                            content = str(result)
                    except Exception:
                        content = str(result)
                except Exception as e:
                    content = f"Tool '{tool_name}' failed: {e}"

                session.dossier["log"].append({
                    "name": tool_name,
                    "status": "approved",
                    "time": ts
                })

            session.messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            session.pending_calls.pop(0)
            session.hops += 1

            if not session.pending_calls:
                run_agent_step(session)
            
            return session.to_dict()

        raise HTTPException(status_code=400, detail="Invalid action requested.")

    except Exception as e:
        # Clear critical state, append error message as an assistant turn, and return updated state
        session.pending_calls = []
        session.hops = 0
        session.awaiting_location = False
        error_msg = f"Aisle encountered an error: {e}. Please check your keys/network and try again."
        session.chat_log.append({
            "role": "assistant",
            "content": error_msg,
            "structured": None
        })
        session.messages.append(AIMessage(content=error_msg))
        return session.to_dict()

@app.get("/api/geoip")
def api_geoip():
    city = get_ip_city()
    return {"city": city}

# Only mount static files and register root index if running locally with static directory present.
# On Vercel, static files and routing are handled on the edge via vercel.json.
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        try:
            with open("static/index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            return HTMLResponse(content="<h3>Frontend assets compiling. Please reload in a moment.</h3>", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
