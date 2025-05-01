from google.adk.agents import Agent
from google.adk.tools import google_search


APP_NAME="google_search_agent"

root_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.5-flash-preview-04-17",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    # google_search is a pre-built tool which allows the agent to perform Google searches.
    tools=[google_search],
    output_key="google_search"
)