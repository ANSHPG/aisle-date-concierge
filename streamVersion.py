"""
AISLE — an AI date-planning concierge
Single-file Streamlit UI wired to a LangChain tool-calling agent.

Run with:  streamlit run app.py
Needs a .env with: OPENAI_API_KEY, TAVILY_API_KEY, OPENWEATHER_API_KEY
"""

import os
import re
import json
import time
import html as html_lib
import requests
import streamlit as st
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate

try:
    from langchain_tavily import TavilySearch
    TAVILY_IMPORT_OK = True
except Exception:
    TAVILY_IMPORT_OK = False


# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + FONTS + THEME
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Aisle · AI Date Concierge", page_icon="✦", layout="wide")

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root{
  --bg:        #0D0D12;
  --bg-raise:  #16161D;
  --card:      #1B1B23;
  --line:      #2C2C36;
  --ink:       #F3EFE8;
  --ink-dim:   #A6A2B0;
  --ink-faint: #6C687A;
  --rose:      #C9184A;
  --rose-soft: #E64A72;
  --gold:      #D4AF37;
  --sage:      #7FA98E;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2.2rem; padding-bottom: 6rem; max-width: 1180px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
  background: var(--bg-raise);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container{ padding-top: 2rem; }

.dossier-eyebrow{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  color: var(--rose-soft);
  text-transform: uppercase;
  margin-bottom: 0.15rem;
}
.dossier-title{
  font-family:'Fraunces', serif;
  font-size: 1.5rem;
  font-weight: 500;
  color: var(--ink);
  margin: 0 0 0.9rem 0;
}
.dossier-rule{
  border: none;
  border-top: 1px dashed var(--line);
  margin: 1.1rem 0;
}
.dossier-label{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 0.3rem;
}
.dossier-readout{
  font-family:'Fraunces', serif;
  font-size: 1.9rem;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.1;
}
.dossier-sub{
  font-family:'Inter', sans-serif;
  font-size: 0.82rem;
  color: var(--ink-dim);
  margin-top: 0.15rem;
}
.status-row{ display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; color:var(--ink-dim); margin-bottom:0.4rem;}
.dot{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.dot-on{ background: var(--sage); box-shadow: 0 0 6px var(--sage); }
.dot-off{ background: var(--rose); box-shadow: 0 0 6px var(--rose); }

.log-entry{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.74rem;
  color: var(--ink-dim);
  border-left: 2px solid var(--line);
  padding: 0.15rem 0 0.15rem 0.6rem;
  margin-bottom: 0.55rem;
}
.log-entry b{ color: var(--ink); font-weight: 600; }
.log-approved{ border-left-color: var(--sage); }
.log-denied{ border-left-color: var(--rose); }

/* ---------- Header ---------- */
.hero-eyebrow{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--rose-soft);
  margin-bottom: 0.4rem;
}
.hero-title{
  font-family:'Fraunces', serif;
  font-size: 3.1rem;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.05;
  margin: 0;
}
.hero-title em{ color: var(--gold); font-style: italic; }
.hero-sub{
  font-family:'Inter', sans-serif;
  font-size: 0.98rem;
  color: var(--ink-dim);
  max-width: 46ch;
  margin-top: 0.7rem;
}
.hero-rule{ border:none; border-top:1px solid var(--line); margin: 1.6rem 0 1.8rem 0; }

/* ---------- Quick prompts ---------- */
div[data-testid="stHorizontalBlock"] .stButton button{
  background: transparent;
  border: 1px solid var(--line);
  color: var(--ink-dim);
  border-radius: 999px;
  font-size: 0.8rem;
  padding: 0.35rem 0.95rem;
  transition: all 0.15s ease;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover{
  border-color: var(--rose-soft);
  color: var(--ink);
}

/* ---------- Chat bubbles ---------- */
.msg-row{ display:flex; margin-bottom: 1.1rem; }
.msg-row.user{ justify-content: flex-end; }
.msg-row.assistant{ justify-content: flex-start; }

.bubble{
  max-width: 72%;
  padding: 0.85rem 1.1rem;
  border-radius: 14px;
  font-size: 0.95rem;
  line-height: 1.55;
  white-space: pre-wrap;
}
.bubble.user{
  background: linear-gradient(155deg, var(--rose) 0%, #9C1339 100%);
  color: #FBEAEE;
  border-bottom-right-radius: 3px;
}
.bubble.assistant{
  background: var(--card);
  border: 1px solid var(--line);
  color: var(--ink);
  border-bottom-left-radius: 3px;
}
.bubble-label{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin: 0 0.3rem 0.3rem 0.3rem;
}

/* ---------- Copy buttons ---------- */
.copy-row{ display:flex; margin-top: 0.35rem; }
.copy-row.user{ justify-content: flex-end; }
.copy-row.assistant{ justify-content: flex-start; }
.copy-btn{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: transparent;
  border: 1px solid var(--line);
  color: var(--ink-faint);
  border-radius: 999px;
  padding: 0.2rem 0.65rem;
  cursor: pointer;
}
.copy-btn:hover{ border-color: var(--rose-soft); color: var(--ink); }

/* ---------- Tool approval card ---------- */
.approval-card{
  background: var(--bg-raise);
  border: 1px solid var(--gold);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin: 0.6rem 0 1.2rem 0;
}
.approval-eyebrow{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.35rem;
}
.approval-title{ font-family:'Fraunces', serif; font-size: 1.15rem; color: var(--ink); margin-bottom: 0.25rem;}
.approval-args{ font-family:'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--ink-dim); }

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"]{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
}
div[data-testid="stChatInput"] textarea{ color: var(--ink) !important; }

/* generic buttons */
.stButton button{ border-radius: 8px; }

/* ---------- Result hierarchy (recommendations + plan) ---------- */
.assistant-extra{ max-width: 72%; margin-top: 0.6rem; }

.rec-wrap{ margin-top: 0.2rem; }
.rec-heading{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.6rem;
}
a.rec-top-card, a.rec-card{ text-decoration: none; color: inherit; }
.rec-top-card{
  display:block;
  background: linear-gradient(155deg, rgba(212,175,55,0.14), rgba(201,24,74,0.08));
  border: 1px solid var(--gold);
  border-radius: 14px;
  padding: 1.05rem 1.3rem;
  margin-bottom: 0.85rem;
  transition: transform 0.15s ease, border-color 0.15s ease;
}
.rec-top-card:hover{ transform: translateY(-2px); border-color: var(--rose-soft); }
.rec-top-tag{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.35rem;
}
.rec-top-name{ font-family:'Fraunces', serif; font-size: 1.35rem; font-weight: 600; color: var(--ink); margin-bottom: 0.3rem; }
.rec-top-desc{ font-family:'Inter', sans-serif; font-size: 0.88rem; color: var(--ink-dim); line-height: 1.5; }

.rec-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.65rem; }
.rec-card{
  display:block;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.8rem 0.95rem;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.rec-card:hover{ border-color: var(--rose-soft); transform: translateY(-2px); }
.rec-card-name{ font-family:'Fraunces', serif; font-size: 1.0rem; font-weight: 600; color: var(--rose-soft); margin-bottom: 0.2rem; }
.rec-card-desc{ font-family:'Inter', sans-serif; font-size: 0.79rem; color: var(--ink-dim); line-height: 1.45; }

.plan-wrap{ margin-top: 0.4rem; }
.plan-block{ margin: 0.75rem 0; }
.plan-label{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 0.25rem;
}
.plan-value{ font-family:'Inter', sans-serif; font-size: 0.92rem; color: var(--ink); line-height: 1.55; }

/* ---------- Timeline ---------- */
.timeline-list{
  position: relative;
  margin: 0.35rem 0 0.4rem 0;
  padding-left: 1.2rem;
  border-left: 2px solid var(--line);
}
.timeline-item{ position: relative; margin-bottom: 0.95rem; }
.timeline-item:last-child{ margin-bottom: 0; }
.timeline-dot{
  position: absolute;
  left: -1.56rem;
  top: 0.2rem;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--gold);
  box-shadow: 0 0 6px var(--gold);
}
.timeline-time{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.74rem;
  letter-spacing: 0.06em;
  color: var(--gold);
  margin-bottom: 0.15rem;
}
.timeline-activity{
  font-family:'Inter', sans-serif;
  font-size: 0.9rem;
  color: var(--ink);
  line-height: 1.5;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# One-time bridge that lets any ".copy-btn" in the app copy its `data-copy`
# attribute to the clipboard. Delegated on the parent document so it keeps
# working across reruns without needing a JS framework.
st.iframe(
    srcdoc="""
    <script>
    (function() {
        if (window.parent.__aisleCopyListenerAttached) { return; }
        window.parent.__aisleCopyListenerAttached = true;
        window.parent.document.addEventListener('click', function(e) {
            var btn = e.target.closest('.copy-btn');
            if (!btn) { return; }
            var text = btn.getAttribute('data-copy') || '';
            var nav = window.parent.navigator;
            if (nav.clipboard && nav.clipboard.writeText) {
                nav.clipboard.writeText(text).then(function() {
                    var original = btn.innerText;
                    btn.innerText = 'Copied ✓';
                    setTimeout(function() { btn.innerText = original; }, 1200);
                }).catch(function() {});
            }
        }, true);
    })();
    </script>
    """,
    height=1,
    scrolling=False,
)


# ────────────────────────────────────────────────────────────────────────────
# BACKEND — tools, prompts, agent wiring (from the provided logic)
# ────────────────────────────────────────────────────────────────────────────

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

MAX_TOOL_HOPS = 6

RESTAURANT_KEYWORDS = [
    "restaurant", "restaurants", "cafe", "café", "coffee shop", "eatery",
    "eateries", "food", "eat", "dining", "brunch", "dinner spot",
    "date spot", "date spots", "places to eat", "bistro",
]


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model="gpt-5-mini")


@st.cache_resource(show_spinner=False)
def get_tavily():
    if not TAVILY_IMPORT_OK or not TAVILY_KEY:
        return None
    return TavilySearch(max_results=4)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for the given city"""
    if not WEATHER_KEY:
        return "Weather tool unavailable: OPENWEATHER_API_KEY is not set."
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": WEATHER_KEY, "units": "metric"}
    response = requests.get(url, params=params, timeout=15)
    data = response.json()
    if response.status_code != 200:
        return f"Error {response.status_code}: {data}"
    st.session_state.dossier["city"] = data.get("name", city)
    st.session_state.dossier["weather"] = {
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "desc": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
    }
    return (
        f"The current weather in {data['name']} is "
        f"{data['weather'][0]['description']} with a temperature of "
        f"{data['main']['temp']}°C, feels like {data['main']['feels_like']}°C, "
        f"humidity at {data['main']['humidity']}%, and wind speed "
        f"{data['wind']['speed']} m/s."
    )


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
    results = tavily.invoke(query)
    if not results:
        return f"No recent news found for {city}."
    summaries = []
    for result in results.get("results", []):
        title = result.get("title", "No title")
        content = result.get("content", "No details")
        summaries.append(f"{title}: {content}")
    return "\n-----------------------\n".join(summaries) if summaries else f"No recent news found for {city}."


def get_coordinates(city: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "my-date-planner-agent"}
    response = requests.get(url, params=params, headers=headers, timeout=15)
    data = response.json()
    if not data:
        return None
    return {"lat": data[0]["lat"], "lon": data[0]["lon"]}


def reverse_geocode(lat, lon):
    """Turn device coordinates into a city name using OpenStreetMap Nominatim."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {"User-Agent": "my-date-planner-agent"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        addr = data.get("address", {})
        return (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("county")
            or addr.get("state")
        )
    except Exception:
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
    st.session_state.dossier["city"] = city
    return city


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


# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []  # list of dicts: {role, content, structured?}
    if "pending_calls" not in st.session_state:
        st.session_state.pending_calls = []
    if "need_agent_step" not in st.session_state:
        st.session_state.need_agent_step = False
    if "dossier" not in st.session_state:
        st.session_state.dossier = {"city": None, "weather": None, "log": []}
    if "hops" not in st.session_state:
        st.session_state.hops = 0
    if "thinking" not in st.session_state:
        st.session_state.thinking = False
    if "location_saved" not in st.session_state:
        st.session_state.location_saved = False
    if "awaiting_location" not in st.session_state:
        st.session_state.awaiting_location = False
    if "pending_user_text" not in st.session_state:
        st.session_state.pending_user_text = ""


init_state()


# ────────────────────────────────────────────────────────────────────────────
# FINAL-ANSWER PARSING → STRUCTURED RESULT
# ────────────────────────────────────────────────────────────────────────────

def _extract_json_object(text: str):
    """Best-effort extraction of a JSON object from an LLM response."""
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


def parse_final_json(text: str):
    """Turn the model's raw output into a normalized structured dict, or None on failure."""
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return None

    message = str(data.get("message") or "").strip()
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else None

    city = (st.session_state.dossier.get("city") or "").strip()
    restaurants = []
    for r in data.get("restaurants") or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        desc = str(r.get("why") or r.get("description") or "").strip()
        query = f"{name} {city}".strip()
        link = f"https://www.google.com/maps/search/{quote(query)}"
        restaurants.append({"name": name, "desc": desc, "link": link})

    return {"message": message, "plan": plan, "restaurants": restaurants}


# ────────────────────────────────────────────────────────────────────────────
# AGENT STEP LOGIC
# ────────────────────────────────────────────────────────────────────────────

def run_agent_step():
    """Ask the LLM for the next move. Either it wants tools, or it's done."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))
    response = llm_with_tools.invoke(st.session_state.messages)
    st.session_state.messages.append(response)

    if response.tool_calls and st.session_state.hops < MAX_TOOL_HOPS:
        st.session_state.pending_calls = list(response.tool_calls)
        st.session_state.need_agent_step = False
    else:
        finalize_response()


def finalize_response():
    llm = get_llm()
    final = llm.invoke(FINAL_PROMPT.format(messages=st.session_state.messages))
    raw = final.content
    structured = parse_final_json(raw)
    display_text = structured["message"] if structured and structured.get("message") else raw

    st.session_state.messages.append(AIMessage(content=raw))
    st.session_state.chat_log.append(
        {"role": "assistant", "content": display_text, "structured": structured}
    )
    st.session_state.pending_calls = []
    st.session_state.need_agent_step = False
    st.session_state.hops = 0
    st.session_state.thinking = False


def execute_tool_call(tool_call, approved: bool):
    tool_name = tool_call["name"]
    ts = time.strftime("%H:%M:%S")

    if not approved:
        content = f"Tool call '{tool_name}' was denied by the user. Proceed without this information."
        st.session_state.dossier["log"].append(
            {"name": tool_name, "status": "denied", "time": ts}
        )
    else:
        try:
            result = TOOLS[tool_name].invoke(tool_call["args"])
            content = str(result)
        except Exception as e:
            content = f"Tool '{tool_name}' failed: {e}"
        st.session_state.dossier["log"].append(
            {"name": tool_name, "status": "approved", "time": ts}
        )

    st.session_state.messages.append(
        ToolMessage(content=content, tool_call_id=tool_call["id"])
    )
    st.session_state.pending_calls.pop(0)
    st.session_state.hops += 1

    if not st.session_state.pending_calls:
        st.session_state.need_agent_step = True


def submit_user_message(text: str):
    """Route the message either straight to the agent, or through a one-time
    location request when it's about restaurants/food and no city is known yet."""
    st.session_state.chat_log.append({"role": "user", "content": text})

    needs_location = False
    if mentions_restaurant_intent(text):
        mentioned_city = quick_extract_city(text)
        if mentioned_city:
            # user explicitly named a city — use it, and remember it going forward
            st.session_state.dossier["city"] = mentioned_city
            st.session_state.location_saved = True
        elif not st.session_state.location_saved:
            # first time asking about restaurants/food with no city known — ask for location
            needs_location = True

    if needs_location:
        st.session_state.awaiting_location = True
        st.session_state.pending_user_text = text
        return

    st.session_state.messages.append(HumanMessage(content=text))
    st.session_state.hops = 0
    st.session_state.thinking = True
    st.session_state.need_agent_step = True


def resume_after_location(city):
    """Called once a city has been resolved from device location."""
    pending = st.session_state.pop("pending_user_text", "")
    st.session_state.dossier["city"] = city
    st.session_state.location_saved = True
    note_text = (
        f"{pending}\n\n(System note: the user's current city was detected as {city} "
        f"via device location. Use this as the city for this request — do not ask for "
        f"the city again unless the user names a different one later.)"
    )
    st.session_state.messages.append(HumanMessage(content=note_text))
    st.session_state.hops = 0
    st.session_state.thinking = True
    st.session_state.need_agent_step = True


def reset_session():
    for key in [
        "messages", "chat_log", "pending_calls", "need_agent_step", "dossier",
        "hops", "thinking", "location_saved", "awaiting_location", "pending_user_text",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()


# ────────────────────────────────────────────────────────────────────────────
# GEOLOCATION CALLBACK (via query params, set by a small JS snippet below)
# ────────────────────────────────────────────────────────────────────────────

_qp = dict(st.query_params)

if st.session_state.awaiting_location and _qp.get("geo_lat") and _qp.get("geo_lon"):
    detected_city = reverse_geocode(_qp["geo_lat"], _qp["geo_lon"])
    st.session_state.awaiting_location = False
    st.query_params.clear()
    if detected_city:
        resume_after_location(detected_city)
    else:
        st.session_state.pending_user_text = ""
        st.session_state.chat_log.append({
            "role": "assistant",
            "content": "I couldn't quite pin down your city from that location — mind typing it in?",
            "structured": None,
        })
    st.rerun()

elif st.session_state.awaiting_location and _qp.get("geo_error"):
    st.session_state.awaiting_location = False
    st.session_state.pending_user_text = ""
    st.query_params.clear()
    st.session_state.chat_log.append({
        "role": "assistant",
        "content": "No worries — I couldn't access your location. Just tell me which city you'd like restaurant ideas for!",
        "structured": None,
    })
    st.rerun()


# NOTE: the actual agent call is deliberately made further down, AFTER the
# chat log (including the user's just-sent message) has already been rendered.
# Streamlit streams each element to the browser as soon as it's drawn, so
# rendering the user's bubble first means it appears instantly, instead of
# only showing up once the (slower) LLM/tool response is ready.


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — THE DOSSIER
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="dossier-eyebrow">Case file · 001</div>', unsafe_allow_html=True)
    st.markdown('<div class="dossier-title">The Dossier</div>', unsafe_allow_html=True)

    st.markdown('<div class="dossier-label">System status</div>', unsafe_allow_html=True)
    checks = [
        ("OpenAI (planning brain)", bool(OPENAI_KEY)),
        ("OpenWeather (climate)", bool(WEATHER_KEY)),
        ("Tavily (city intel)", bool(TAVILY_KEY) and TAVILY_IMPORT_OK),
    ]
    for label, ok in checks:
        dot = "dot-on" if ok else "dot-off"
        st.markdown(
            f'<div class="status-row"><span class="dot {dot}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="dossier-rule">', unsafe_allow_html=True)

    st.markdown('<div class="dossier-label">Target city</div>', unsafe_allow_html=True)
    city = st.session_state.dossier.get("city")
    st.markdown(f'<div class="dossier-readout">{city or "—"}</div>', unsafe_allow_html=True)

    weather = st.session_state.dossier.get("weather")
    if weather:
        st.markdown('<div class="dossier-label" style="margin-top:0.9rem;">Conditions</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dossier-readout">{weather["temp"]}°C</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dossier-sub">{weather["desc"]} · feels {weather["feels_like"]}°C · '
            f'{weather["humidity"]}% humidity · wind {weather["wind"]} m/s</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="dossier-rule">', unsafe_allow_html=True)
    st.markdown('<div class="dossier-label">Tool activity</div>', unsafe_allow_html=True)

    log = st.session_state.dossier.get("log", [])
    if not log:
        st.markdown('<div class="dossier-sub">No tools called yet.</div>', unsafe_allow_html=True)
    else:
        for entry in reversed(log[-8:]):
            cls = "log-approved" if entry["status"] == "approved" else "log-denied"
            st.markdown(
                f'<div class="log-entry {cls}"><b>{entry["name"]}</b><br>{entry["status"]} · {entry["time"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="dossier-rule">', unsafe_allow_html=True)
    if st.button("↺  Start a new date", use_container_width=True):
        reset_session()
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# MAIN — HERO
# ────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-eyebrow">AI Date Concierge</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Every good date<br>starts with <em>good intel.</em></h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Tell Aisle the city, the vibe, or the occasion. '
    'It reads the weather, the local news, and the map before it commits to a plan — '
    'and it asks before it acts.</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="hero-rule">', unsafe_allow_html=True)

if not st.session_state.chat_log:
    st.markdown('<div class="dossier-label" style="margin-bottom:0.6rem;">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    quick_prompts = [
        "Plan a romantic evening in Paris",
        "Quick rainy-day date idea in Mumbai",
        "Full itinerary for a first date in Rome",
        "Best cafe for a low-key coffee date in Pune",
    ]
    for c, p in zip(cols, quick_prompts):
        with c:
            if st.button(p, use_container_width=True, key=f"qp_{p}"):
                submit_user_message(p)
                st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# MAIN — CHAT LOG
# ────────────────────────────────────────────────────────────────────────────

PLAN_FIELDS = [
    ("timeline", "Best Date Plan"),
    ("best_area", "Best Area / Location"),
    ("best_time", "Best Time"),
    ("things_to_avoid", "Things To Avoid"),
    ("backup_plan", "Backup Plan"),
]
PLAN_LABELS = dict(PLAN_FIELDS)


def render_timeline_html(timeline):
    """Render the plan's timeline as a vertical, dotted timeline instead of a paragraph."""
    if not timeline:
        return ""
    items_html = []

    if isinstance(timeline, list):
        for step in timeline:
            if isinstance(step, dict):
                t = str(step.get("time") or "").strip()
                a = str(step.get("activity") or step.get("description") or "").strip()
            else:
                t, a = "", str(step).strip()
            if not a:
                continue
            time_html = f'<div class="timeline-time">{t}</div>' if t else ""
            items_html.append(
                f'<div class="timeline-item"><span class="timeline-dot"></span>'
                f'{time_html}<div class="timeline-activity">{a}</div></div>'
            )
    else:
        # Fallback for a plain-text timeline: split on inline clock-time markers.
        text = str(timeline).strip()
        pattern = r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
        pieces = re.split(pattern, text)
        if len(pieces) > 1:
            buf_time = ""
            for piece in pieces:
                piece = piece.strip(" -–—:")
                if not piece:
                    continue
                if re.fullmatch(pattern, piece):
                    buf_time = piece
                else:
                    time_html = f'<div class="timeline-time">{buf_time}</div>' if buf_time else ""
                    items_html.append(
                        f'<div class="timeline-item"><span class="timeline-dot"></span>'
                        f'{time_html}<div class="timeline-activity">{piece}</div></div>'
                    )
                    buf_time = ""
        else:
            items_html.append(
                f'<div class="timeline-item"><span class="timeline-dot"></span>'
                f'<div class="timeline-activity">{text}</div></div>'
            )

    if not items_html:
        return ""
    return '<div class="timeline-list">' + "".join(items_html) + '</div>'


def build_assistant_copy_text(entry):
    """Flatten an assistant turn (message + plan + restaurants) into plain text for copying."""
    parts = [entry.get("content") or ""]
    structured = entry.get("structured")
    if structured:
        restaurants = structured.get("restaurants") or []
        if restaurants:
            parts.append("\nRecommended places:")
            for r in restaurants:
                line = f"- {r.get('name', '')}"
                if r.get("desc"):
                    line += f": {r['desc']}"
                parts.append(line)

        plan = structured.get("plan")
        if plan:
            parts.append("\nDate plan:")
            timeline = plan.get("timeline")
            if timeline:
                parts.append(f"{PLAN_LABELS['timeline']}:")
                if isinstance(timeline, list):
                    for step in timeline:
                        if isinstance(step, dict):
                            t = step.get("time", "")
                            a = step.get("activity") or step.get("description") or ""
                            parts.append(f"  {t} — {a}".strip(" —"))
                        else:
                            parts.append(f"  {step}")
                else:
                    parts.append(f"  {timeline}")
            for key, label in PLAN_FIELDS:
                if key == "timeline":
                    continue
                val = plan.get(key)
                if val:
                    parts.append(f"{label}: {val}")
    return "\n".join(p for p in parts if p is not None)


def render_assistant_entry(entry):
    """Render an assistant turn as a bubble, followed by a designed hierarchy
    of recommendation cards and/or plan sections when structured data exists."""
    parts = ['<div class="msg-row assistant"><div style="max-width:72%;">']
    parts.append('<div class="bubble-label">Aisle</div>')
    parts.append(f'<div class="bubble assistant">{entry["content"]}</div>')

    structured = entry.get("structured")
    if structured:
        restaurants = structured.get("restaurants") or []
        plan = structured.get("plan")

        if restaurants:
            top, rest = restaurants[0], restaurants[1:]
            parts.append('<div class="assistant-extra"><div class="rec-wrap">')
            parts.append('<div class="rec-heading">✦ Curated for you</div>')
            parts.append(f'<a class="rec-top-card" href="{top["link"]}" target="_blank" rel="noopener">')
            parts.append('<div class="rec-top-tag">Top Pick</div>')
            parts.append(f'<div class="rec-top-name">{top["name"]}</div>')
            if top["desc"]:
                parts.append(f'<div class="rec-top-desc">{top["desc"]}</div>')
            parts.append('</a>')
            if rest:
                parts.append('<div class="rec-grid">')
                for r in rest:
                    parts.append(f'<a class="rec-card" href="{r["link"]}" target="_blank" rel="noopener">')
                    parts.append(f'<div class="rec-card-name">{r["name"]}</div>')
                    if r["desc"]:
                        parts.append(f'<div class="rec-card-desc">{r["desc"]}</div>')
                    parts.append('</a>')
                parts.append('</div>')
            parts.append('</div></div>')

        if plan:
            parts.append('<div class="assistant-extra"><div class="plan-wrap"><hr class="dossier-rule">')
            timeline = plan.get("timeline")
            if timeline:
                parts.append(
                    f'<div class="plan-block"><div class="plan-label">{PLAN_LABELS["timeline"]}</div>'
                    f'{render_timeline_html(timeline)}</div>'
                )
            for key, label in PLAN_FIELDS:
                if key == "timeline":
                    continue
                val = plan.get(key)
                if val:
                    parts.append(
                        f'<div class="plan-block"><div class="plan-label">{label}</div>'
                        f'<div class="plan-value">{val}</div></div>'
                    )
            parts.append('</div></div>')

    copy_text = build_assistant_copy_text(entry)
    escaped_copy = html_lib.escape(copy_text, quote=True)
    parts.append(
        f'<div class="assistant-extra"><div class="copy-row assistant">'
        f'<button class="copy-btn" data-copy="{escaped_copy}">⧉ Copy</button></div></div>'
    )

    parts.append('</div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


for entry in st.session_state.chat_log:
    if entry["role"] == "user":
        escaped_user = html_lib.escape(entry["content"], quote=True)
        st.markdown(
            f'<div class="msg-row user"><div>'
            f'<div class="bubble-label" style="text-align:right;">You</div>'
            f'<div class="bubble user">{entry["content"]}</div>'
            f'<div class="copy-row user"><button class="copy-btn" data-copy="{escaped_user}">⧉ Copy</button></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        render_assistant_entry(entry)

# Location-request card (fires once per session, only for restaurant/food asks with no known city)
if st.session_state.awaiting_location:
    st.markdown(
        '''<div class="approval-card">
              <div class="approval-eyebrow">One quick thing</div>
              <div class="approval-title">Aisle would like your location</div>
              <div class="approval-args">To recommend nearby restaurants, Aisle needs to know your city.
              Share your device location, or just type your city in the chat box below.</div>
            </div>''',
        unsafe_allow_html=True,
    )
    st.iframe(
        srcdoc="""
        <div style="font-family:Inter,sans-serif;display:flex;align-items:center;">
          <button id="aisleGeoBtn" style="background:linear-gradient(155deg,#C9184A,#9C1339);color:#FBEAEE;
            border:none;padding:0.55rem 1.1rem;border-radius:8px;font-size:0.82rem;cursor:pointer;
            font-family:Inter,sans-serif;">
            ✦ Share my location
          </button>
          <span id="aisleGeoStatus" style="margin-left:0.6rem;font-size:0.78rem;color:#A6A2B0;
            font-family:'JetBrains Mono',monospace;"></span>
        </div>
        <script>
        document.getElementById('aisleGeoBtn').addEventListener('click', function () {
          var status = document.getElementById('aisleGeoStatus');
          if (!navigator.geolocation) {
            window.parent.location.href = window.parent.location.pathname + "?geo_error=1";
            return;
          }
          status.innerText = 'Locating…';
          navigator.geolocation.getCurrentPosition(function (pos) {
            var lat = pos.coords.latitude;
            var lon = pos.coords.longitude;
            window.parent.location.href = window.parent.location.pathname + "?geo_lat=" + lat + "&geo_lon=" + lon;
          }, function () {
            window.parent.location.href = window.parent.location.pathname + "?geo_error=1";
          }, {
            enableHighAccuracy: false,
            timeout: 6000,
            maximumAge: 300000
          });
        });
        </script>
        """,
        height=55,
        scrolling=False,
    )

# Pending tool approval card
if st.session_state.pending_calls:
    call = st.session_state.pending_calls[0]
    args_str = ", ".join(f"{k}={v!r}" for k, v in call.get("args", {}).items())
    st.markdown(
        f'''<div class="approval-card">
              <div class="approval-eyebrow">Awaiting your approval</div>
              <div class="approval-title">Aisle wants to call <code>{call["name"]}</code></div>
              <div class="approval-args">{args_str or "no arguments"}</div>
            </div>''',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("✓  Approve", key=f"approve_{call['id']}", use_container_width=True):
            execute_tool_call(call, approved=True)
            st.rerun()
    with c2:
        if st.button("✕  Deny", key=f"deny_{call['id']}", use_container_width=True):
            execute_tool_call(call, approved=False)
            st.rerun()

elif st.session_state.need_agent_step:
    # The user's message (and this indicator) are already drawn on screen at
    # this point — only now do we make the actual, slower LLM/tool call.
    st.markdown(
        '<div class="dossier-sub" style="font-family:\'JetBrains Mono\',monospace;">'
        '· · · Aisle is thinking</div>',
        unsafe_allow_html=True,
    )
    with st.spinner(""):
        run_agent_step()
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# MAIN — INPUT
# ────────────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Tell Aisle about your date… e.g. \"plan a rooftop dinner in Mumbai this weekend\"")
if user_input and not st.session_state.pending_calls:
    # If the user types instead of sharing location, cancel the location ask
    # and just process their new message normally.
    st.session_state.awaiting_location = False
    st.session_state.pending_user_text = ""
    submit_user_message(user_input)
    st.rerun()