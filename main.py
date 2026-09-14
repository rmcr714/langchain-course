import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from tavily import TavilyClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


load_dotenv()
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    
)

tools = [TavilySearch()]

agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in tokyo?")]})
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
