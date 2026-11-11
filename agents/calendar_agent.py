from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain_google_community import CalendarToolkit

load_dotenv()

class CalendarAgent:
    def __init__(self, current_location='Ho Chi Minh City/vietnam'):
        # Google Calendar tools
        toolkit = CalendarToolkit()
        tools = toolkit.get_tools()

        # Web search tool
        tools.append(
            TavilySearch(
                max_results=5,
                topic="news",
                search_depth="advanced",
            )
        )

        # Language model
        model = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
        )

        # Agent that can call the tools
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                f"You help users plan outings using their calendar and current news. "
                "Use calendar tools to check availability and resolve relative dates. "
                "Use web search for current conditions at the destination. "
                "Include source URLs and dates when reporting news. "
                "Explain uncertainty when evidence is insufficient. "
                "Only modify calendar events when explicitly requested. "
                "Use {current_location} for the user's timezone unless specified otherwise."
            ),
        )

    def forward(self, request):
        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Check my calendar for tonight and a few days ahead. "
                            "Check if I have tasks, deadlines, assignments, exams, "
                            "or personal commitments that could affect going out tonight. "
                            "Mention any relevant disruptions. "
                            "Is it alright to go out tonight? Explain why."
                        )
                    }
                ]
            }
        )

        return result["messages"][-1].content

