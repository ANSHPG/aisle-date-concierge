from dotenv import load_dotenv
load_dotenv(override=True)

import os
import requests

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")

from langchain.tools import tool
# @tool

from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_tavily import TavilySearch

#weather tool
api_key = os.getenv("OPENWEATHER_API_KEY")
@tool
def get_weather(city:str) -> str:
    """Get the current weather for the given city"""
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        return f"Error {response.status_code}: {data}"

    return (
        f"The current weather in {data['name']} is "
        f"{data['weather'][0]['description']} with a temperature of "
        f"{data['main']['temp']}°C, feels like {data['main']['feels_like']}°C, "
        f"humidity at {data['main']['humidity']}%, and wind speed "
        f"{data['wind']['speed']} m/s."
    )

#Tavily news tool
tavily = TavilySearch(max_results=4)
@tool
def get_news(city: str) -> str:
    """Get latest important city updates for planning a date like traffic, road closures, crime, events, and popular restaurants."""

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

    return "\n-----------------------\n".join(summaries)


from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Extract the city name only from the following prompt: {prompt}")
prompt2 = PromptTemplate.from_template("""
You are an intelligent AI date planning assistant.

Below is the full conversation history, including:
- User preferences
- City information
- Weather data
- Local news
- Restaurant suggestions

Conversation:
{messages}

Your job is to generate the best response based on the user's latest intent.

Decision rules:

1. First understand what the user wants:
   - If they ask for a simple suggestion (restaurant, cafe, area, quick idea), give a short and concise answer.
   - If they ask for a complete plan, itinerary, or full evening plan, give a detailed structured answer.

2. Use context smartly:
   - Use weather only if it affects the plan.
   - Use local news only if safety, traffic, or public events matter.
   - Use restaurant options only when relevant.

Planning rules:
- Prefer indoor plans during bad weather.
- Prefer outdoor options only when weather is comfortable.
- Avoid unsafe or crowded areas if news mentions them.
- Avoid unnecessary long travel.
- Match suggestions to budget, vibe, and convenience.
- Prioritize romance, comfort, practicality, and realism.

Response style:
- Keep answers short by default (2–5 lines).
- Only expand when the user explicitly wants a full plan.
- Do not repeat tool outputs unnecessarily.
- Be direct, helpful, and natural.

If the user asks for a full plan, return in this format:

1. Best Date Plan
(Complete timeline)

2. Best Restaurant Recommendation
(Name + why)

3. Best Area / Location
(Why this area)

4. Best Time
(When and why)

5. Things To Avoid
(Traffic, weather, safety, budget mistakes)

6. Backup Plan
(Alternative option)

Tone:
Warm, concise, romantic, practical, and personalized.
""")

#coordinates
def get_coordinates(city: str):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "my-date-planner-agent"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if not data:
        return None

    return {
        "lat": data[0]["lat"],
        "lon": data[0]["lon"]
    }

#City Tool
@tool
def get_city(user_input: str) -> str:
    """Extract the city name from the user's prompt"""
    response = llm.invoke(prompt.format(prompt=user_input))
    return response.content.strip()

#Get Restaurants Tool
@tool
def get_restaurants(city: str) -> str:
    """Get nearby restaurants in the given city using OpenStreetMap Overpass API"""

    coords = get_coordinates(city)

    if not coords:
        return f"Could not find coordinates for {city}"

    lat = coords["lat"]
    lon = coords["lon"]

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
        "https://z.overpass-api.de/api/interpreter"
    ]

    headers = {
        "User-Agent": "my-date-planner-agent"
    }

    for url in overpass_urls:
        try:
            response = requests.get(
                url,
                params={"data": query},
                headers=headers,
                timeout=15
            )

            if response.status_code != 200:
                continue

            try:
                data = response.json()
            except Exception:
                continue

            restaurants = []
            seen = set()

            for place in data.get("elements", []):
                tags = place.get("tags", {})
                name = tags.get("name")

                if name and name not in seen:
                    seen.add(name)
                    restaurants.append(name)

            if restaurants:
                return "\n".join(restaurants[:10])

        except requests.exceptions.Timeout:
            continue

        except requests.exceptions.RequestException:
            continue

    return f"Restaurant lookup failed for {city}. Try again later."
    
    
tools = {
    "get_weather": get_weather,
    "get_news": get_news,
    "get_city": get_city,
    "get_restaurants": get_restaurants
}

llm_pwrd_tool = llm.bind_tools([get_weather, get_news, get_city, get_restaurants])


#Agent Loop

messages = [
    SystemMessage(content="""
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

Response style:

- Keep responses short, natural, and conversational by default.
- Only give detailed structured plans when the user explicitly asks for:
  - a full plan
  - itinerary
  - complete date plan
  - plan my evening
- For simple questions, answer in 2–5 lines.
- For recommendations, keep it concise and practical.
- Expand only when necessary.

Planning logic:

- Prefer indoor plans during bad weather.
- Avoid unsafe or high-traffic areas if news mentions them.
- Match recommendations to budget, vibe, and convenience.
- Prioritize comfort, romance, practicality, and realism.

Tone:
Warm, smart, concise, helpful, and natural.
""")
]

print("Welcome to the AI Agent!\nType 'exit' to quit.")

while True:
    user_input = input("You:")
    if user_input.lower() == "exit":
        print("Exiting the AI Agent. Goodbye!")
        break
    messages.append(HumanMessage(content=user_input))

    while True:
        response = llm_pwrd_tool.invoke(messages)
        messages.append(response)

        #if tool is required

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]

                #human in the loop
                confirmation = input(f"Agent wants to call {tool_name}\n Approve (yes/no)?")
                if confirmation.lower() == "no":
                    print(f"Tool call {tool_name} denied, so i can not fetch the latest information")
                    break

                #execute the tool
                tool_message = tools[tool_name].invoke(tool_call)
                messages.append(ToolMessage(
                    content = tool_message,
                    tool_call_id = tool_call["id"]
                ))

            continue
        else:
            break
    ai_response = llm.invoke(prompt2.format(messages=messages))
    print("Agent:", ai_response.content)
    messages.append(ai_response)



