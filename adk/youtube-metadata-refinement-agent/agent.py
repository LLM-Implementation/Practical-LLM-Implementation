# --- YouTube Metadata Multi-Agent Refinement Workflow ---
# This script defines a multi-agent system using the Google Agent Development Kit (ADK).
# The workflow aims to improve YouTube video metadata (title, description, tags)
# by orchestrating several LLM agents (Gemini, GPT, Claude, etc.) through distinct phases:
# 1. Greeter: Initializes the process and fetches initial data.
# 2. Criticism (Parallel): Multiple LLMs critique the initial metadata.
# 3. Refinement (Parallel): Multiple LLMs refine the metadata based on all critiques.
# 4. Voting (Parallel): Multiple LLMs vote on the refinements (excluding their own).
# 5. Aggregation: A custom agent aggregates votes and determines the final ranking.
# It utilizes ADK components like LlmAgent, BaseAgent, SequentialAgent, ParallelAgent,
# LiteLlm for multi-model support, FunctionTools for specific actions, and SessionState
# for communication between agents.
# ---

import logging
import os
import asyncio
import json
import re
from typing import AsyncGenerator, List, Dict, Any, Optional, Union
from typing_extensions import override
from collections import defaultdict
from pydantic import BaseModel, Field
import pathlib

# --- ADK Imports ---
from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool, BaseTool
from google.adk.tools.tool_context import ToolContext

# --- Local Utils Import ---
from . import utils # Assuming utils.py is in the same directory

# --- Configure Logging (File + Console) ---
LOG_FILE = "agent_workflow.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s' # Added function name/line

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Set root logger level to DEBUG

# Clear existing handlers (important if running in interactive environments like Jupyter)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Create Formatter
formatter = logging.Formatter(LOG_FORMAT)

# Create Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Keep console less verbose unless needed
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Create File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE, mode='w') # 'w' overwrites file each run, 'a' appends
    file_handler.setLevel(logging.DEBUG) # Log DEBUG and above to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    print(f"--- Logging configured. Output will be sent to console (INFO+) and file '{LOG_FILE}' (DEBUG+) ---")
except Exception as e:
    print(f"--- WARNING: Failed to configure file logging to '{LOG_FILE}'. Error: {e} ---")
    print("--- Logging will proceed only to console. ---")


# Configure ADK logging level (optional, affects verbosity in file/console)
logging.getLogger('google.adk').setLevel(logging.DEBUG) # Set ADK's logger to DEBUG

# Get logger for this specific module
logger = logging.getLogger(__name__) # Use module's name for specific logs

# --- Constants ---
APP_NAME = "youtube_multi_refine_app"
USER_ID = "multi_llm_user"
SESSION_ID = "multi_llm_session_002" # Changed session ID slightly

# Model Names
MODEL_GEMINI_FLASH = "gemini/gemini-2.5-flash-preview-04-17"
MODEL_GEMINI_PRO = "gemini/gemini-2.5-pro-preview-03-25"
MODEL_GPT_4O = "openai/gpt-4o" 
MODEL_CLAUDE_SONNET = "anthropic/claude-3-7-sonnet-20250219"
# MODEL_GPT_4_MINI = "openai/gpt-4.1-mini"
# MODEL_CLAUDE_HAIKU = "anthropic/claude-3-5-haiku-20241022"
MODEL_DEEPSEEK_CHAT = "deepseek/deepseek-chat"
MODEL_GROK_3_BETA = "xai/grok-3-beta"
GREETER_MODEL = "gemini-2.5-flash-preview-04-17"


# State Keys (Consistent Naming)
STATE_KEY_INPUT_TITLE = "input_title"
STATE_KEY_INPUT_DESC = "input_description"
STATE_KEY_INPUT_TAGS = "input_tags"
STATE_KEY_INPUT_DOCS = "supporting_docs"
STATE_KEY_GREETING = "greeting_message"
STATE_KEY_CRITICISM_SUFFIX = "_criticism"
STATE_KEY_REFINEMENT_SUFFIX = "_refinement"
STATE_KEY_VOTE_SUFFIX = "_vote_result"
STATE_KEY_FINAL_RANKING = "final_ranking_result"

# --- Instruction Loading ---
PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
GREETER_INSTRUCTION = utils.load_instruction(PROMPTS_DIR / "greeter_instruction.txt")
CRITIC_INSTRUCTION = utils.load_instruction(PROMPTS_DIR / "critic_instruction.txt")
REFINER_INSTRUCTION = utils.load_instruction(PROMPTS_DIR / "refiner_instruction.txt")
VOTER_INSTRUCTION = utils.load_instruction(PROMPTS_DIR / "voter_instruction.txt")

# --- Initial Metadata Input ---
initial_metadata = {
    STATE_KEY_INPUT_TITLE: "Google Agent Development Kit (ADK) Quickstart: Build Your First Gemini AI Agent",
    STATE_KEY_INPUT_DESC: """🚀 Get started with the new Google Agent Development Kit (ADK)! In this Python quickstart tutorial, we'll walk you through building your very first AI agent powered by Google's Gemini models.\n\nWhat is the Google Agent Development Kit (ADK)?\nADK is an open-source, flexible, and modular framework from Google designed for developing and deploying AI agents, with tight integration into the Google ecosystem and Gemini models.\n\nIn this video, you'll learn:\n*   Setting up your Python environment (venv) for ADK development.\n*   Installing the google-adk package using pip.\n*   Creating the basic project structure for your agent.\n*   Writing the core agent code (agent.py) including tool definitions (get_weather, get_current_time).\n*   Understanding the agent configuration (using Gemini 1.5 Flash model).\n*   Setting up your Google AI Studio API key in the .env file.\n*   Running the agent locally using the ADK Dev UI (`adk web`).\n*   Interacting with your first agent and seeing function calling in action!\n\nWe cover key concepts like agent tools, function calls, and basic prompt instructions within the ADK framework.\n\n🔗 Useful Links:\n*   Official Google ADK Documentation & GitHub: https://google.github.io/adk-docs/\n\n⏱️ Timestamps:\n00:00 - Intro to Google Agent Development Kit (ADK)\n00:36 - Quickstart Overview & Requirements\n00:50 - Step 1: Setup Environment & Install ADK (venv, pip install google-adk)\n01:03 - Step 2: Create Agent Project Structure (multi_tool_agent folder)\n01:14 - Creating __init__.py and agent.py\n01:26 - Agent.py Code Walkthrough (Imports, Tools: get_weather, get_current_time)\n01:47 - Defining the Root Agent (Name, Model: Gemini 1.5 Flash, Description, Instructions, Tools)\n02:22 - Creating the .env file\n02:30 - Step 3: Set up the Model (Google AI Studio API Key)\n02:48 - Getting the API Key from Google AI Studio\n03:00 - Adding API Key to .env file\n03:14 - Step 4: Run Your Agent (adk web & Dev UI)\n03:39 - Interacting with the Agent in the Dev UI\n03:50 - Demo: Asking for Weather (Function Call: get_weather)\n04:01 - Exploring Events in Dev UI\n04:24 - Demo: Asking for Time (Function Call: get_current_time)\n04:46 - Demo: Handling Variations (NYC vs New York) & Simple Prompt Engineering\n05:30 - Example: Tweaking Instructions for Better Recognition\n05:52 - Final Test with Updated Instructions\n06:07 - Wrap Up & Next Steps (Multi-Agent Teaser)\n\n👍 If you found this tutorial helpful, please hit the like button and subscribe for more AI agent content, including the upcoming video on multi-agent interactions with ADK!\n💬 Let me know in the comments what you think of Google ADK!\n\n#GoogleADK #GeminiAI #Python #AIAgent #GoogleAI #AgentDevelopmentKit #Tutorial #FunctionCalling #LLM""",
    STATE_KEY_INPUT_TAGS: "Google Agent Development Kit, Google ADK, ADK Python, Gemini, Gemini AI, Google Gemini, AI Agent, Artificial Intelligence, Python, Python Tutorial, AI Agent Framework, Google AI, Google AI Studio, Function Calling, Agent Tools, Build AI Agent, ADK Quickstart, ADK Tutorial, Open Source AI, Large Language Model, LLM, VS Code, Programming, Coding, Software Development, pip install google-adk, adk web",
    STATE_KEY_INPUT_DOCS: """
## Video Analysis\n\nMain Topics:\n\nIntroduction to the Google Agent Development Kit (ADK).\n\nADK Quickstart guide walkthrough.\n\nEnvironment setup for ADK (Python virtual environment, pip installation).\n\nCreating an ADK agent project structure and necessary files (__init__.py, agent.py, .env).\n\nDefining agent tools (Python functions like get_weather, get_current_time).\n\nConfiguring the agent (defining name, model, description, instructions, tools).\n\nSetting up API credentials (using Google AI Studio API Key).\n\nRunning the ADK agent using the provided web UI (adk web).\n\nInteracting with the agent via the web UI, demonstrating function calling.\n\nModifying agent instructions to handle variations in user input (e.g., \"nyc\" vs \"New York\").\n\nKey Information/Takeaways:\n\nADK is an open-source AI agent framework from Google, integrated with Gemini and Google tools.\n\nIt allows developing and deploying AI agents.\n\nThe Quickstart guide provides steps to create a basic agent that uses tools (functions) to answer questions about weather and time.\n\nADK requires Python (3.9+ recommended) and environment setup.\n\nAgents are defined in Python scripts (agent.py) specifying model, tools, and instructions.\n\nAPI keys (e.g., from Google AI Studio) are needed for the LLM interaction and stored in an .env file.\n\nADK includes a development web UI (adk web) for easy testing and interaction.\n\nThe agent can interpret user queries, decide which tool (function) to call, extract parameters (like city names), and generate responses based on the tool's output.\n\nAgent instructions (prompts) can be customized to improve how the agent understands and handles user input variations.\n\nOverall Summary:\nThis video serves as a practical tutorial demonstrating how to get started with Google's Agent Development Kit (ADK). The presenter follows the official Quickstart guide, showing viewers how to set up their environment, create a simple multi-tool agent project, configure it with tools and instructions, add API credentials, and run/interact with the agent using the built-in web UI. The example agent can retrieve weather and time information for specified cities, and the video also shows how to modify the agent's instructions to improve its understanding of varied city name inputs.\n\nTone and Style:\n\nTone: Informative, instructional, practical, and generally clear. The presenter seems knowledgeable and guides the viewer through the process systematically.\n\nStyle: Screen recording walkthrough/tutorial. The style is hands-on, showing the exact commands typed and the resulting UI/code changes.\n\nTarget Audience:\nThe video is primarily intended for developers, particularly those familiar with Python, who are interested in building AI agents using Google's technologies like Gemini and the new Agent Development Kit. Viewers should have some comfort with command-line interfaces, IDEs (like VS Code), and basic concepts of AI/LLMs and APIs.\n\nStructure:\n\nIntroduction: Briefly introduce ADK and its purpose, state the goal of following the Quickstart.\n\nEnvironment Setup: Creating a virtual environment and installing ADK.\n\nProject Creation: Creating the necessary folder structure and Python/environment files.\n\nAgent Coding: Copying/pasting and explaining the code for __init__.py and agent.py (defining tools and the agent itself).\n\nConfiguration: Creating the .env file and explaining how to get/add the Google AI Studio API key.\n\nRunning the Agent: Using the adk web command to start the development server and UI.\n\nInteraction & Demonstration: Using the web UI to interact with the agent, ask questions, and observe the function calling mechanism and responses.\n\nModification & Improvement: Showing how to modify agent instructions to handle input variations (\"nyc\") and re-testing.\n\nConclusion: Summarizing the Quickstart completion, mentioning future topics (multi-agent), thanking the audience, and a call to subscribe.\n\n## Video Transcript\n\nHey everyone, welcome back to my channel. So, I think recently the Google just released an agent development kit, and it's an open source agent framework with the Gemini and Google. What is the agent development kit? It's for developing and deploying AI agents. I'm very interested in the multi-agent functionalities, like how do agents interact with each other? But before we're doing that, we're going to first get familiar with the agent development kit, and we're going to do some quick start.\n\nLike it recommends a local IDE like VS Code, which I'm going to be using with Python 3.9 and terminal access to run this quick application. So, let's go ahead and do that.\n\nFirst, we're going to set up the environment and install the ADK. What we will do is to create a virtual environment. Okay, we're going to install this Google ADK.\n\nNext, we're going to create our agent project. So the agent project will have the following structures. We're going to first create the multi tool agent folder. Next, we're going to type this command and try to write the code into the file.\n\nNext, we're going to create an agent.py file in the same folder.\n\nNext, we're going to copy and paste the following code into the agent.py script. So, before we're doing that, let's see what's going on here. So, it's importing the agent methods from the ADK. So basically it defines two functions, the get weather and get current time. And this part is the agent part. And what we're going to do is to have the name weather_time_agent, have the model Gemini 2.0 flash, and we have the description here. Agent to answer the question about time and weather in a city. And we push some instructions there. You are a helpful agent who can answer the user questions about the time and weather in a city. And also we define the tools here like the get_weather, get_current_time.\n\nNext, we're going to create the environment file in the same folder. So we create a .env file. And then, so, next, we're going to set up the model. So there's two options here using the Google AI Studio API key or the Google Cloud Vertex AI API key. I think the studio will be more straightforward, very simple way. Let's try it. Okay, so well, in this page, you can see that you can just create an API key and search a project, select that and create an API key in existing project. So we're going to copy paste the following arguments to the environment file. Okay, so you can see that it has, it tells the Google Gen AI whether to use the vertex AI. You just paste the key there.\n\nAlright. Next, we're going to run our agent. So, make sure that we are navigating to the parent director of the agent project. There are multiple ways to interact with that. I'm very interested in the ADK UI tool they developed. So let's just run this. So we can open the URL, usually something like this. Okay, cool. And then we just select the multi tool agent, and then we can just type anything here.\n\nSo let's see what kind of questions we can type. So, for example, check the weather in New York. It's get the weather in New York with the temperature 25 degrees Celsius. And it also can check the event. You can check different events and see the exactly what's happening here. So it's going to basically tells whether time agent determines the get weather function tool to process this query. And you can see the city arguments in the function call is got the New York City. How about time there? All right. We can also check that the weather time agent decides to use the get current time tool. And the arguments it recognizes New York. So, once the the tool gets processed, the response will be the current time in there, which is showing here.\n\nHow about this? Let's try some tricky one to see if that can handle the issue like um how about weather in NYC? Weather information for NYC is not available. So it somehow can have some feedback loop to ask users to make sure that you are asking the right questions. You can see that the event it recognized the NYC, but it's not, as you remember in our function, it's hard coded, so it won't recognize the NYC. But somehow it asks users to confirm that if it means like New York. And once I confirmed, it just somehow feed in the New York to the function call and get the response.\n\nSo actually, you know, from the agent, we can do tweak some instructions to let the model to recognize those variations. For example, please recognize the variations of city names and convert it to correct full name.\n\nLet's see if this going to work. Let's directly try this example NYC. I can now just recognize the NYC, get the response. How about New York City? All right. Time there? It also writes the time in the New York City.\n\nAll right. So that's the first agent using ADK. Congratulations. I think the next step we're going to do some advanced tutorial on the multi-agent things. Thanks for watching and hope you enjoy it. And please don't forget to subscribe to support the channel. And see you in the next video.
"""
}

# --- Create LiteLlm Instances ---
# Initialize LLM wrappers based on available API keys
llm_instances: Dict[str, LiteLlm] = {}
available_providers = []

try:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and not gemini_key.startswith("YOUR_"):
        llm_instances["gemini"] = LiteLlm(model=MODEL_GEMINI_PRO)
        available_providers.append("gemini")
        logger.info(f"✅ LiteLlm instance created for Gemini: {MODEL_GEMINI_PRO}")
    else:
        logger.warning("⚠️ Skipping LiteLlm for Gemini: API Key not set or is placeholder.")
except Exception as e:
    logger.error(f"❌ Failed to create LiteLlm for Gemini: {e}. Check API Key.", exc_info=True)

# --- Use Gemini as placeholder for GPT and Claude if keys aren't available ---
# This allows the structure to remain but use only Gemini if others aren't configured
gpt_llm = LiteLlm(model=MODEL_GEMINI_FLASH)# Default to Gemini if OpenAI key missing
claude_llm = LiteLlm(model=MODEL_GEMINI_FLASH) # Default to Gemini if Anthropic key missing
deepseek_llm = LiteLlm(model=MODEL_GEMINI_FLASH) # Default to Gemini if DeepSeek key missing
xai_llm = LiteLlm(model=MODEL_GEMINI_FLASH) # Default to Gemini if XAI key missing

try:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("YOUR_"):
        gpt_llm = LiteLlm(model=MODEL_GPT_4O)
        llm_instances["gpt"] = gpt_llm
        available_providers.append("gpt")
        logger.info(f"✅ LiteLlm instance created for OpenAI: {MODEL_GPT_4O}")
    else:
        logger.warning("⚠️ Skipping LiteLlm for OpenAI: API Key not set or is placeholder. Using Gemini for GPT agents if available.")
        if gpt_llm: llm_instances["gpt"] = gpt_llm # Assign Gemini if available
except Exception as e:
    logger.error(f"❌ Failed to create LiteLlm for OpenAI: {e}. Check API Key.", exc_info=True)


try:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key and not anthropic_key.startswith("YOUR_"):
        claude_llm = LiteLlm(model=MODEL_CLAUDE_SONNET)
        llm_instances["claude"] = claude_llm
        available_providers.append("claude")
        logger.info(f"✅ LiteLlm instance created for Anthropic: {MODEL_CLAUDE_SONNET}")
    else:
        logger.warning("⚠️ Skipping LiteLlm for Anthropic: API Key not set or is placeholder. Using Gemini for Claude agents if available.")
        if claude_llm: llm_instances["claude"] = claude_llm # Assign Gemini if available
except Exception as e:
    logger.error(f"❌ Failed to create LiteLlm for Anthropic: {e}. Check API Key.", exc_info=True)

try:
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key and not deepseek_key.startswith("YOUR_"):
        deepseek_llm = LiteLlm(model=MODEL_DEEPSEEK_CHAT)
        llm_instances["deepseek"] = deepseek_llm
        available_providers.append("deepseek")
        logger.info(f"✅ LiteLlm instance created for DeepSeek: {MODEL_DEEPSEEK_CHAT}")
    else:
        logger.warning("⚠️ Skipping LiteLlm for DeepSeek: API Key not set or is placeholder. Using Gemini for DeepSeek agents if available.")
        if deepseek_llm: llm_instances["deepseek"] = deepseek_llm # Assign Gemini if available and key is missing
except Exception as e:
    logger.error(f"❌ Failed to create LiteLlm for DeepSeek: {e}. Check API Key.", exc_info=True)

try:
    xai_key = os.environ.get("XAI_API_KEY")
    if xai_key and not xai_key.startswith("YOUR_"):
        xai_llm = LiteLlm(model=MODEL_GROK_3_BETA)
        llm_instances["xai"] = xai_llm
        available_providers.append("xai")
        logger.info(f"✅ LiteLlm instance created for XAI: {MODEL_GROK_3_BETA}")
    else:
        logger.warning("⚠️ Skipping LiteLlm for XAI: API Key not set or is placeholder. Using Gemini for XAI agents if available.")
        if xai_llm: llm_instances["xai"] = xai_llm # Assign Gemini if available and key is missing
except Exception as e:
    logger.error(f"❌ Failed to create LiteLlm for XAI: {e}. Check API Key.", exc_info=True)


# --- Fallback LLM ---
fallback_llm = llm_instances.get("gemini") # Prefer Gemini as fallback
if not fallback_llm and llm_instances:
    first_available_provider = next(iter(llm_instances))
    fallback_llm = llm_instances[first_available_provider]
    logger.warning(f"⚠️ Gemini LLM not available, using {fallback_llm.model} ({first_available_provider}) as fallback.")
elif not llm_instances:
     logger.critical("❌ No LLM instances could be initialized. Cannot proceed. Check API Key configurations and network.")
     raise RuntimeError("No LLM instances could be initialized.")


# --- Tools ---

# Tool 1: Get Initial Metadata
def get_initial_metadata(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieves initial metadata from configuration and ALSO saves it
    to the session state using predefined keys.
    """
    logger.info("--- Tool: get_initial_metadata called ---")
    state = tool_context.state

    if 'initial_metadata' not in globals():
        logger.error("Tool: initial_metadata global variable not found!")
        # Return an error dict, but also don't attempt to update state
        return {"error": "Initial metadata configuration missing."}

    # Fetch from global config (assuming this is intended)
    title = initial_metadata.get(STATE_KEY_INPUT_TITLE, "Title not found in config")
    description = initial_metadata.get(STATE_KEY_INPUT_DESC, "Desc not found in config")
    tags = initial_metadata.get(STATE_KEY_INPUT_TAGS, "Tags not found in config")
    supporting_docs = initial_metadata.get(STATE_KEY_INPUT_DOCS, "Docs not found in config")

    # *** ADDED: Update the session state directly via tool_context ***
    state_updates = {
        STATE_KEY_INPUT_TITLE: title,
        STATE_KEY_INPUT_DESC: description,
        STATE_KEY_INPUT_TAGS: tags,
        STATE_KEY_INPUT_DOCS: supporting_docs,
    }
    # The ADK framework handles appending an event with this state_delta
    # when the tool returns, associating the state change with the tool call.
    # We use tool_context.state.update() which signals the desired delta.
    # Note: Directly assigning like state[KEY]=value might not work reliably
    # for triggering the state delta mechanism within a tool context.
    # Using update() on the state object within the context is safer.
    try:
        tool_context.state.update(state_updates)
        logger.info(f"--- Tool: Updated session state with keys: {list(state_updates.keys())}")
    except Exception as e:
        logger.error(f"--- Tool: Failed to update session state. Error: {e}", exc_info=True)
        # Return the error in the response as well
        return {"error": f"Failed to update session state: {e}"}


    # Return the fetched metadata as the tool's direct result as well
    metadata_to_return = {
        "title": title,
        "description": description,
        "tags": tags,
        "supporting_docs": supporting_docs
    }
    logger.info(f"--- Tool: Returning initial metadata and signaled state update.")
    return metadata_to_return

# Tool 2: Get Metadata and Criticisms (for Refiners)
def get_metadata_and_criticisms(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieves initial metadata and all parsed criticisms from session state."""
    tool_name = "get_metadata_and_criticisms"
    logger.info(f"--- Tool: {tool_name} called ---")
    state = tool_context.state
    criticisms = {}
    # *** CORRECT FIX: Iterate directly over state, which yields keys ***
    criticism_keys = [k for k in state.to_dict().keys() if k.endswith(STATE_KEY_CRITICISM_SUFFIX)] # Reverted to direct iteration
    logger.info(f"--- Tool {tool_name}: Found potential criticism keys via direct iteration: {criticism_keys}")

    for key in criticism_keys: # Iterate over the collected keys
        value_to_parse = state.get(key) # Use .get() for safe access using the key
        # Use the imported utility function
        parsed_criticism = utils._parse_json_string(value_to_parse, context_key=f"{tool_name}:{key}")
        if parsed_criticism is not None:
            criticisms[key] = parsed_criticism
        else:
            raw_value_snippet = str(value_to_parse)[:100] + "..." if value_to_parse else "None"
            logger.warning(f"Tool {tool_name}: Failed to parse criticism from state key '{key}'. Raw value snippet: {raw_value_snippet}")
            criticisms[key] = {"error": f"Could not parse criticism in state key '{key}'"}

    # Fetch initial metadata parts as well (no changes needed here)
    data = {
        "title": state.get(STATE_KEY_INPUT_TITLE, "Title not found in state."),
        "description": state.get(STATE_KEY_INPUT_DESC, "Description not found in state."),
        "tags": state.get(STATE_KEY_INPUT_TAGS, "Tags not found in state."),
        "supporting_docs": state.get(STATE_KEY_INPUT_DOCS, "Supporting docs not found in state."),
        "criticisms": criticisms
    }
    for key in ["title", "description", "tags", "supporting_docs"]:
        if "not found" in str(data.get(key, "")):
             logger.warning(f"Tool '{tool_name}': Key '{key}' was not found in session state.")

    logger.info(f"--- Tool {tool_name}: Returning data for refinement (Found {len(criticisms)} valid/error criticisms).")
    return data

# Tool 3: Get Refinements and Criticisms for Voting (for Voters)
def get_refinements_and_criticisms_for_voting(
    tool_context: ToolContext, voter_agent_name: str
) -> Dict[str, Any]:
    """Retrieves all criticisms and refinements (excluding the voter's own corresponding refinement) from session state."""
    tool_name = "get_refinements_and_criticisms_for_voting"
    logger.info(f"--- Tool: {tool_name} called by {voter_agent_name} ---")
    state = tool_context.state
    all_data = {}

    # 1. Get Criticisms (Parsed)
    criticisms = {}
    # *** CORRECT FIX: Iterate directly over state, which yields keys ***
    criticism_keys = [k for k in state.to_dict().keys() if k.endswith(STATE_KEY_CRITICISM_SUFFIX)] # Reverted to direct iteration
    logger.info(f"--- Tool {tool_name}: Found potential criticism keys via direct iteration: {criticism_keys}")
    for key in criticism_keys: # Iterate over collected keys
        value_to_parse = state.get(key) # Use .get() for safe access
        # Use the imported utility function
        parsed_criticism = utils._parse_json_string(value_to_parse, context_key=f"{tool_name}:Criticism:{key}")
        if parsed_criticism is not None:
            criticisms[key] = parsed_criticism
        else:
            raw_value_snippet = str(value_to_parse)[:100] + "..." if value_to_parse else "None"
            logger.warning(f"Tool {tool_name}: Failed to parse criticism from state key '{key}'. Raw value snippet: {raw_value_snippet}")
            criticisms[key] = {"error": f"Could not parse criticism in state key '{key}'"}
    all_data["criticisms"] = criticisms

    # 2. Get Refinements (Parsed, excluding self)
    # *** CORRECT FIX: Iterate directly over state, which yields keys ***
    refinement_keys = [k for k in state.to_dict().keys() if k.endswith(STATE_KEY_REFINEMENT_SUFFIX)] # Reverted to direct iteration
    refinements_to_rank = {}

    # ... (voter exclusion logic remains the same) ...
    voter_provider_match = re.match(r"(\w+)Voter", voter_agent_name, re.IGNORECASE)
    if not voter_provider_match:
         logger.error(f"Tool {tool_name}: Could not extract provider from voter name '{voter_agent_name}'. Cannot exclude self.")
         all_data["refinements_to_rank"] = {"error": f"Could not identify voter provider from '{voter_agent_name}' to exclude self."}
         return all_data

    voter_provider_lower = voter_provider_match.group(1).lower()
    excluded_refiner_key = f"{voter_provider_lower}{STATE_KEY_REFINEMENT_SUFFIX}"
    logger.info(f"--- Tool {tool_name}: Voter is '{voter_agent_name}'. Excluding refinement key: '{excluded_refiner_key}'.")

    found_exclusion_key = False
    for key in refinement_keys: # Iterate over collected keys
        if key == excluded_refiner_key:
            found_exclusion_key = True
            logger.debug(f"--- Tool {tool_name}: Excluding self-refinement '{key}' for ranking by '{voter_agent_name}'.")
            continue

        value_to_parse = state.get(key) # Use .get() for safe access
        # Use the imported utility function
        parsed_refinement = utils._parse_json_string(value_to_parse, context_key=f"{tool_name}:Refinement:{key}")
        if parsed_refinement is not None:
            logger.debug(f"--- Tool {tool_name}: Including refinement '{key}' for ranking by '{voter_agent_name}'.")
            refinements_to_rank[key] = parsed_refinement
        else:
            raw_value_snippet = str(value_to_parse)[:100] + "..." if value_to_parse else "None"
            logger.warning(f"Tool {tool_name}: Could not parse refinement in state key '{key}', excluding it from ranking. Raw value snippet: {raw_value_snippet}")
            refinements_to_rank[key] = {"error": f"Could not parse refinement in state key '{key}'"}

    # ... (rest of the function remains the same) ...
    all_data["refinements_to_rank"] = refinements_to_rank
    logger.info(f"--- Tool {tool_name}: Returning data for {voter_agent_name}. Ranking {len(refinements_to_rank)} valid/error refinements).")
    return all_data

# Wrap functions as ADK Tools
get_initial_metadata_tool = FunctionTool(func=get_initial_metadata)
get_metadata_and_criticisms_tool = FunctionTool(func=get_metadata_and_criticisms)
get_refinements_for_voting_tool = FunctionTool(func=get_refinements_and_criticisms_for_voting)

# --- Agent Definitions ---

# 1. Greeter Agent
greeter = LlmAgent(
    name="Greeter",
    model=GREETER_MODEL,
    instruction=GREETER_INSTRUCTION, # Use loaded instruction
    tools=[get_initial_metadata_tool],
    output_key=STATE_KEY_GREETING
)
logger.info(f"Agent '{greeter.name}' defined.")

# 2. Critic Agents (Run in Parallel)
critics: List[LlmAgent] = []
# critic_instruction loaded from file above

for provider, llm in llm_instances.items():
    critic_name = f"{provider.capitalize()}Critic"
    output_key = f"{provider}{STATE_KEY_CRITICISM_SUFFIX}"
    critics.append(LlmAgent(name=critic_name, model=llm, instruction=CRITIC_INSTRUCTION, tools=[get_initial_metadata_tool], output_key=output_key)) # Use loaded instruction
    logger.info(f"Agent '{critic_name}' defined using model {llm.model if llm else 'None'}.")

if not critics:
    logger.error("❌ No critic LLMs initialized. Workflow may fail.")

# 3. Refiner Agents (Run in Parallel)
refiners: List[LlmAgent] = []
# refiner_instruction loaded from file above

for provider, llm in llm_instances.items():
    refiner_name = f"{provider.capitalize()}Refiner"
    output_key = f"{provider}{STATE_KEY_REFINEMENT_SUFFIX}"
    refiners.append(LlmAgent(name=refiner_name, model=llm, instruction=REFINER_INSTRUCTION, tools=[get_metadata_and_criticisms_tool], output_key=output_key)) # Use loaded instruction
    logger.info(f"Agent '{refiner_name}' defined using model {llm.model if llm else 'None'}.")

if not refiners:
    logger.error("❌ No refiner LLMs initialized. Workflow may fail.")


# 4. Voter Agents (Run in Parallel) - Only if enough refiners exist
voter_agents: List[LlmAgent] = []
if len(refiners) >= 2:
    # voter_instruction loaded from file above
    for provider, llm in llm_instances.items():
        voter_name = f"{provider.capitalize()}Voter"
        output_key = f"{provider}{STATE_KEY_VOTE_SUFFIX}"
        voter_agents.append(LlmAgent(name=voter_name, model=llm, instruction=VOTER_INSTRUCTION, tools=[get_refinements_for_voting_tool], output_key=output_key)) # Use loaded instruction
        logger.info(f"Agent '{voter_name}' defined using model {llm.model if llm else 'None'}.")

    if not voter_agents:
        logger.warning("⚠️ Configured LLMs could not be initialized as voters. Voting phase will be skipped.")
else:
    logger.warning(f"⚠️ Skipping Voting phase definition: Only {len(refiners)} refiner(s) available (requires >= 2).")


# 5. Vote Aggregator Agent (Custom BaseAgent)
class VoteAggregatorAgent(BaseAgent):
    """
    Aggregates votes from voter agents' results stored directly in session state.
    Determines a final ranking based on scores derived from individual ranks.
    Reads state keys ending with STATE_KEY_VOTE_SUFFIX directly.
    """
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str = "VoteAggregator"):
        super().__init__(name=name)
        logger.debug(f"Initialized {self.name}")

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting vote aggregation.")
        state = ctx.session.state
        #
        #This state returned as a dict from the ctx.
        #
        # --- Step 1: Process Votes directly from State ---
        all_votes_data = {}
        # *** CORRECT FIX: Iterate directly over state, which yields keys ***
        vote_keys = [k for k in state.keys() if k.endswith(STATE_KEY_VOTE_SUFFIX)] # Reverted to direct iteration
        logger.info(f"[{self.name}] Found potential vote keys in state via direct iteration: {vote_keys}")

        for key in vote_keys: # Iterate over collected keys
            value_to_parse = state.get(key) # Use .get() for safe access
            # Use the imported utility function
            parsed_vote = utils._parse_json_string(value_to_parse, context_key=f"{self.name}:Vote:{key}")
            # ... (rest of parsing check remains the same) ...
            if isinstance(parsed_vote, dict) and "ranked_refinements" in parsed_vote and isinstance(parsed_vote["ranked_refinements"], list):
                all_votes_data[key] = parsed_vote
                logger.info(f"[{self.name}] Successfully parsed vote from key '{key}'.")
            elif isinstance(parsed_vote, dict) and "error" in parsed_vote:
                 logger.warning(f"[{self.name}] Skipping vote key '{key}' due to error reported during parsing: {parsed_vote.get('error')}")
            else:
                 raw_value_snippet = str(value_to_parse)[:100] + "..." if value_to_parse else "None"
                 logger.warning(f"[{self.name}] Skipping vote key '{key}' due to unknown parsing issue or invalid structure. Raw value snippet: {raw_value_snippet}")

        # --- Step 2: Aggregation Logic ---
        scores = defaultdict(int)
        rank_counts = defaultdict(lambda: defaultdict(int))
        possible_refiner_keys = {k for k in state if k.endswith(STATE_KEY_REFINEMENT_SUFFIX)}
        if not possible_refiner_keys:
             logger.error(f"[{self.name}] No refinement keys found in state. Cannot aggregate votes.")
             yield Event(
                 author=self.name,
                 content=types.Content(parts=[types.Part(text="Aggregation failed: No refinements found in state.")])
            )
             return

        logger.info(f"[{self.name}] Aggregating votes for potential refiners: {possible_refiner_keys}")
        num_voters_with_data = len(all_votes_data)
        expected_rank_count = max(0, len(possible_refiner_keys) - 1)

        if expected_rank_count == 0 and len(possible_refiner_keys) == 1:
             logger.info(f"[{self.name}] Only one refiner found. Ranking is trivial.")
        elif expected_rank_count < 1:
             logger.warning(f"[{self.name}] Calculation error: expected_rank_count is {expected_rank_count} with {len(possible_refiner_keys)} refiners.")

        for vote_key, vote_data in all_votes_data.items():
            voter_name = vote_key.replace(STATE_KEY_VOTE_SUFFIX, '')
            logger.info(f"[{self.name}] Processing vote from: {voter_name}")

            num_ranked = len(vote_data["ranked_refinements"])
            logger.info(f"[{self.name}] {voter_name} ranked {num_ranked} item(s). Expected to rank: {expected_rank_count}")

            if num_ranked != expected_rank_count:
                 logger.warning(f"[{self.name}] Vote from {voter_name} ranked {num_ranked} items, but expected {expected_rank_count}. Ranking might be incomplete or based on filtered data.")

            processed_keys_in_vote = set()
            for item in vote_data["ranked_refinements"]:
                if not isinstance(item, dict) or "rank" not in item or "refiner_key" not in item:
                    logger.warning(f"[{self.name}] Malformed rank item in vote {vote_key}: {item}. Skipping item.")
                    continue

                rank = item.get("rank")
                refiner_key = item.get("refiner_key")

                if not isinstance(rank, int) or rank <= 0:
                     logger.warning(f"[{self.name}] Invalid rank '{rank}' in vote from {voter_name}. Skipping item.")
                     continue
                if not isinstance(refiner_key, str) or not refiner_key.endswith(STATE_KEY_REFINEMENT_SUFFIX):
                    logger.warning(f"[{self.name}] Invalid refiner_key format '{refiner_key}' in vote from {voter_name}. Skipping item.")
                    continue
                if refiner_key not in possible_refiner_keys:
                    logger.warning(f"[{self.name}] Vote from {voter_name} ranks unexpected/missing refiner_key '{refiner_key}'. Skipping item.")
                    continue
                if refiner_key in processed_keys_in_vote:
                     logger.warning(f"[{self.name}] Vote from {voter_name} ranks refiner_key '{refiner_key}' multiple times. Skipping duplicate.")
                     continue
                processed_keys_in_vote.add(refiner_key)

                points = max(0, (expected_rank_count + 1) - rank)
                scores[refiner_key] += points
                rank_counts[refiner_key][rank] += 1
                logger.debug(f"[{self.name}] Score: {refiner_key} (+{points} points for Rank {rank} from {voter_name}) -> Total: {scores[refiner_key]}")

        # --- Step 3: Determine Final Ranking ---
        final_ranking_list = []
        sorted_refiners = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        for i, (refiner_key, score) in enumerate(sorted_refiners):
            final_ranking_list.append({
                "rank": i + 1,
                "refiner_key": refiner_key,
                "score": score,
                "rank_distribution": dict(rank_counts.get(refiner_key, {})),
            })

        accounted_keys = {item['refiner_key'] for item in final_ranking_list}
        for key in possible_refiner_keys:
            if key not in accounted_keys:
                 logger.info(f"[{self.name}] Adding refiner '{key}' with score 0 (received no valid votes/points).")
                 final_ranking_list.append({
                    "rank": len(final_ranking_list) + 1,
                    "refiner_key": key,
                    "score": 0,
                    "rank_distribution": {}
                })

        final_ranking_list.sort(key=lambda x: x['rank'])
        final_result = {"final_ranking": final_ranking_list}
        final_result_json = json.dumps(final_result, indent=2)
        logger.info(f"[{self.name}] Aggregation Complete. Final Ranking:\n{final_result_json}")

        # --- Step 4: Yield final result event with state delta ---
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Vote aggregation complete. Final Ranking:\n{final_result_json}")]),
            actions=EventActions(state_delta={STATE_KEY_FINAL_RANKING: final_result_json}),
        )

# Instantiate the aggregator
aggregator_agent = VoteAggregatorAgent()
logger.info(f"Agent '{aggregator_agent.name}' defined.")


# --- Define Workflow Agents ---

# Phase 1: Criticism (Parallel)
criticism_phase = None
if critics:
    criticism_phase = ParallelAgent(
        name="CriticismPhase",
        sub_agents=critics
    )
    logger.info(f"Workflow Phase '{criticism_phase.name}' defined with {len(critics)} critics.")

# Phase 2: Refinement (Parallel)
refinement_phase = None
if refiners:
    refinement_phase = ParallelAgent(
        name="RefinementPhase",
        sub_agents=refiners
    )
    logger.info(f"Workflow Phase '{refinement_phase.name}' defined with {len(refiners)} refiners.")

# Phase 3: Voting (Parallel)
voting_phase = None
if voter_agents: # Checks if list is non-empty
    voting_phase = ParallelAgent(
        name="VotingPhase",
        sub_agents=voter_agents
    )
    logger.info(f"Workflow Phase '{voting_phase.name}' defined with {len(voter_agents)} voters.")

# Phase 4: Aggregation (Single Agent) is aggregator_agent

# --- Define Root Sequential Workflow Agent ---
root_agent = None # Define variable expected by ADK tools like `adk web`
workflow_steps = []
if greeter: workflow_steps.append(greeter)
if criticism_phase: workflow_steps.append(criticism_phase)
if refinement_phase: workflow_steps.append(refinement_phase)

if voting_phase:
    workflow_steps.append(voting_phase)
    if aggregator_agent:
        workflow_steps.append(aggregator_agent)
        logger.info("Adding Voting and Aggregation steps to the workflow.")
    else:
         logger.error("Aggregator agent not defined, cannot add aggregation step.")
else:
     logger.warning("Voting phase skipped, Aggregation step is also skipped.")

root_agent = SequentialAgent(
        name="MultiModelRefinementWorkflow",
        sub_agents=workflow_steps
    )
logger.info(f"Root Sequential Workflow Agent '{root_agent.name}' created and assigned to 'root_agent'.")


# --- Setup Runner and Session ---
session_service = InMemorySessionService()

# Session Initialization
try:
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_metadata  # Pass initial metadata directly
    )
    logger.info(f"Created new session '{SESSION_ID}' with initial metadata.")
except Exception as e:
    logger.critical(f"❌ Failed to create session: {e}", exc_info=True)
    # In a real app, you might exit or handle this differently
    session = None # Ensure session is None if creation fails

# Runner Initialization
runner = None
if root_agent and session: # Check if root agent and session were created
    try:
        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service
        )
        logger.info(f"Runner initialized with root agent: {root_agent.name}")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize Runner: {e}", exc_info=True)
else:
    if not root_agent: logger.critical("❌ Cannot initialize Runner: root_agent is not defined.")
    if not session: logger.critical("❌ Cannot initialize Runner: session initialization failed.")


# --- Function to Interact with the Agent (Async) ---
async def call_agent_async(runner: Runner, user_id: str, session_id: str):
    """
    Runs the workflow asynchronously and prints the final state.
    """
    if not runner:
        logger.error("Runner is not initialized. Cannot run workflow.")
        return
        
    logger.info(f"Starting simplified workflow run for session '{session_id}'.")
    # Generic starting message
    content = types.Content(role='user', parts=[types.Part(text="Start the metadata refinement workflow.")])

    try:
        # Execute the workflow by iterating through the events
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Optional: Log basic event info for debugging during development
            # logger.debug(f"Event: Author={event.author}, Type={type(event).__name__}, Final={event.is_final_response()}")
            
            # Log phase completions simply
            if criticism_phase and event.author == criticism_phase.name and event.is_final_response():
                 logger.info(f"--- Phase '{event.author}' completed. ---")
            elif refinement_phase and event.author == refinement_phase.name and event.is_final_response():
                 logger.info(f"--- Phase '{event.author}' completed. ---")
            elif voting_phase and event.author == voting_phase.name and event.is_final_response():
                 logger.info(f"--- Phase '{event.author}' completed. ---")
            elif aggregator_agent and event.author == aggregator_agent.name and event.is_final_response():
                 logger.info(f"--- Aggregation Agent '{event.author}' completed. ---")
            # No need to capture final message here, we'll check state

    except Exception as e:
        logger.error(f"Error during workflow execution for session '{session_id}': {e}", exc_info=True)
        print("\n--- Workflow failed with an error. See logs for details. ---")
        return # Stop processing if workflow fails

    # --- Display Final Result ---
    print("\n--- Workflow Execution Complete ---")
    final_session = session_service.get_session(
        app_name=runner.app_name, 
        user_id=user_id,          
        session_id=session_id     
    )
    
    print("\n--- Final Session State ---")
    if final_session:
        final_ranking_json = final_session.state.get(STATE_KEY_FINAL_RANKING)
        if final_ranking_json:
            print(f"**Final Ranking Result (from state key '{STATE_KEY_FINAL_RANKING}')**:")
            try:
                # Try to pretty-print the JSON ranking
                ranking_data = json.loads(final_ranking_json)
                print(json.dumps(ranking_data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("Could not parse ranking JSON, showing raw data:")
                print(final_ranking_json)
            except Exception as e:
                print(f"Error processing ranking data: {e}")
                print(final_ranking_json) # Show raw data on other errors too
        elif voting_phase: # Check if voting phase was supposed to run
             print("Final ranking data not found in state. Aggregation might have failed or skipped.")
        else:
             print("Voting/Aggregation phase was skipped, no ranking expected.")

        print("\n**Full Final State:**")
        # Dump the entire state dictionary for review
        try:
            final_state_dict = final_session.state.to_dict() 
            # Attempt to pretty-print the whole state
            print(json.dumps(final_state_dict, indent=2, ensure_ascii=False)) # Use .to_dict() for Pydantic state proxy
            output_filename = "final_session_state.json"
            logger.info(f"Attempting to write final state to '{output_filename}'...")
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(final_state_dict, f, indent=2, ensure_ascii=False)

                logger.info(f"Successfully wrote final state to '{output_filename}'.")
            except IOError as io_err:
                logger.error(f"Failed to write final state to file '{output_filename}': {io_err}", exc_info=True)

        except Exception as e:
             logger.error(f"Error converting final state to JSON for printing: {e}")
             print(final_session.state) # Print raw state if JSON fails

    else:
        print("Could not retrieve final session state.")
    print("---------------------------------------------------------------------\n")


# --- Simplified Main Execution Block ---
async def main():
    logger.info("--- Starting Main Execution ---")
    
    # Essential Pre-run Checks
    if not llm_instances:
        logger.critical("❌ ERROR: No LLM instances initialized.")
        return
    if not root_agent:
         logger.critical("❌ ERROR: Root workflow agent not defined.")
         return
    if not runner:
         logger.critical("❌ ERROR: Runner not initialized.")
         return
    if not session:
        logger.critical("❌ ERROR: Session not initialized.")
        return

    required_keys = [STATE_KEY_INPUT_TITLE, STATE_KEY_INPUT_DESC, STATE_KEY_INPUT_TAGS, STATE_KEY_INPUT_DOCS]
    if not isinstance(initial_metadata, dict) or not all(k in initial_metadata for k in required_keys):
         logger.critical(f"❌ ERROR: `initial_metadata` incomplete.")
         return
         
    # --- Run Workflow ---
    await call_agent_async(runner, USER_ID, SESSION_ID)
    logger.info("--- Main Execution Finished ---")


if __name__ == "__main__":
    # Keep API Key checks - they are helpful warnings
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") # Accept either
    keys_ok = bool(gemini_key and not gemini_key.startswith("YOUR_"))
    if not keys_ok: logger.warning("⚠️ WARNING: GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set or is placeholder.")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key or openai_key.startswith("YOUR_"): logger.info("INFO: OPENAI_API_KEY not set or is placeholder. GPT agents will use fallback.")
    else: keys_ok = True 

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key or anthropic_key.startswith("YOUR_"): logger.info("INFO: ANTHROPIC_API_KEY not set or is placeholder. Claude agents will use fallback.")
    else: keys_ok = True

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_key or deepseek_key.startswith("YOUR_"): logger.info("INFO: DEEPSEEK_API_KEY not set or is placeholder. DeepSeek agents will use fallback.")
    else: keys_ok = True

    xai_key = os.environ.get("XAI_API_KEY")
    if not xai_key or xai_key.startswith("YOUR_"): logger.info("INFO: XAI_API_KEY not set or is placeholder. XAI agents will use fallback.")
    else: keys_ok = True

    # Check if at least one key is set before running
    if not keys_ok:
         logger.critical("❌ ERROR: No valid API keys found for any provider. Cannot run.")
    else:
        # Run the main async function
        try:
            asyncio.run(main())
        except Exception as e:
             logger.critical(f"❌ An unexpected error occurred during execution: {e}", exc_info=True)

# --- END OF REVISED FILE SECTIONS ---