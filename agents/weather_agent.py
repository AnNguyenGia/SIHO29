import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

load_dotenv()

@tool
def check_weather(location: str) -> dict:
    """Check current weather for a named city, such as Hanoi."""
    try:
# Convert city name to coordinates.
        response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en"},
                timeout=15,
        )
        response.raise_for_status()
        places = response.json().get("results", [])

        if not places:
            return {"status": "unknown", "error": "City not found"}

        place = places[0]

    # Fetch weather at those coordinates.
        response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code,wind_gusts_10m",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                },
                timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        weather = data["current"]
        code = weather["weather_code"]
        gust = weather["wind_gusts_10m"]

        if code is None or gust is None:
            return {"status": "unknown", "error": "Missing weather data"}

        return {
                "status": "ok",
                "location": place["name"],
                "country": place.get("country"),
                "time": weather["time"],
                "timezone": data["timezone"],
                "weather_code": code,
                "wind_gust_kmh": gust,
                "bad_weather": code in (65, 82, 95, 96, 97, 99) or gust >= 50,
                "temperature_c": weather["temperature_2m"],
            }

    except (requests.RequestException, KeyError, ValueError, TypeError) as e:
        return {"status": "unknown", "error": str(e)}



class WeatherAgent:
    def __init__(self):
        self.weather_agent = create_agent(
            model=ChatDeepSeek(model="deepseek-chat", temperature=0),
            tools=[check_weather],
            system_prompt=(
                    "you are agent to check whether weather bad or not"
                    "Use check_weather for weather questions. "
                    "Ask for the city if none is provided; you cannot detect location. "
                    "Report the resolved city, country, and result briefly. "
                    "Unknown means the check failed. "
                    "The bad-weather rules are demo rules, not official warnings."
            ),
        )

    def forward(self, location):
        result = self.weather_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Is there bad weather in {location} now? what is evidence ? explain you think it is bad/not bad weather ?",
            }]
        })
        return result["messages"][-1].content


