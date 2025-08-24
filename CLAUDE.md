# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a practical LLM implementation repository containing tutorials and code examples for fine-tuning various Large Language Models. It serves as a companion repository for the LLM Implementation YouTube channel (www.youtube.com/@LLMImplementation).

## Key Architecture Components

### 1. Google Agent Development Kit (ADK) Projects

**Location**: `adk/` directory

**Core Components**:
- **Agent Definition Pattern**: All agents follow the ADK pattern with `agent.py` as the main entry point
- **Tool Integration**: Agents use pre-built ADK tools (e.g., `google_search`) and custom function tools
- **Multi-Agent Architecture**: Complex workflows use sequential and parallel agent orchestration
- **Session Management**: State persistence across agent interactions using ADK's session system

**Key Files**:
- `adk/google-search-agent/agent.py`: Simple search agent using `google_search` tool
- `adk/youtube-metadata-refinement-agent/agent.py`: Complex multi-agent workflow with criticism → refinement → voting → aggregation phases
- `adk/youtube-metadata-refinement-agent/utils.py`: Shared utilities for JSON parsing and instruction loading

### 2. Agent Framework Examples

**Location**: `agents_frameworks/` directory

Contains Jupyter notebooks demonstrating different agent frameworks:
- LangGraph implementations
- AutoGen (ag2) with OpenAI integration

### 3. Model-Specific Fine-Tuning Examples

**Gemini** (`gemini/`):
- Video analysis demos using Gemini 2.5 Pro
- Supervised fine-tuning workflows with Vertex AI
- Prompt rewriting and optimization tutorials

**Llama** (`llama/`):
- Fine-tuning implementations and tutorials

**GPT** (`gpt-oss-20b/`):
- Open-source model fine-tuning examples

### 4. Context Engineering Framework

**Location**: `context-engineering/` directory

**Key Concept**: 2-step process for building AI features:
1. Setup examples in `examples/` folder (2-5 high-quality, relevant code examples)
2. Use the universal magic prompt template

**Magic Prompt Pattern**:
```
Please scan the project files. The goal is to set up the context engineering environment for a [FEATURE DESCRIPTION] using [FRAMEWORK] — following the /examples/[REFERENCE-EXAMPLE] as a reference, especially its use of [KEY TOOL].
```

## Common Development Commands

### ADK Projects

**Setup Environment**:
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install google-adk -q
pip install python-dotenv -q
pip install litellm -q
```

**Configure API Keys**:
Create `.env` file in `adk/` directory:
```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
XAI_API_KEY=your_xai_api_key
```

**Run ADK Web UI**:
```bash
cd adk/
adk web
```
Access at `http://localhost:8000`

### Jupyter Notebooks

**Install Dependencies**:
```bash
pip install langgraph langchain langsmith
pip install jupyter notebook
```

**Run Notebooks**:
```bash
jupyter notebook
# Navigate to agents_frameworks/ or other notebook directories
```

## Development Patterns

### ADK Agent Structure

**Basic Agent Pattern**:
```python
from google.adk.agents import Agent
from google.adk.tools import tool_name

root_agent = Agent(
    name="agent_name",
    model="gemini-2.5-flash-preview-04-17",
    description="Agent description",
    instruction="Agent instructions",
    tools=[tool_list],
    output_key="output_key"
)
```

**Multi-Agent Workflow Pattern**:
```python
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent

# Define individual agents
agents = [agent1, agent2, agent3]

# Sequential execution
sequential_workflow = SequentialAgent(
    name="WorkflowName", 
    sub_agents=agents
)

# Parallel execution  
parallel_workflow = ParallelAgent(
    name="ParallelWorkflow",
    sub_agents=agents
)
```

### State Management

**Session State Keys**:
- Use consistent naming patterns for state keys
- Suffix patterns: `_criticism`, `_refinement`, `_vote_result`
- Access via `tool_context.state.get(key)` and `tool_context.state.update(dict)`

### Error Handling

**LLM Provider Fallbacks**:
- Multiple LLM providers configured with fallback to Gemini
- API key validation with placeholder detection
- Graceful degradation when providers unavailable

### Logging Configuration

**Multi-Level Logging**:
- Console handler (INFO level)
- File handler (DEBUG level)
- Structured logging with function names and line numbers
- Module-specific loggers

## Project Structure Conventions

```
project_name/
├── agent.py              # Main agent definition
├── __init__.py          # Python package initialization
├── utils.py             # Shared utilities
├── prompts/             # Instruction templates
│   ├── agent_instruction.txt
│   └── tool_instruction.txt
├── requirements.txt     # Python dependencies
└── .env                # API keys (gitignored)
```

## API Key Management

- Store all API keys in `.env` files
- Use placeholder detection to avoid errors with unconfigured keys
- Support multiple LLM providers with intelligent fallback
- Never commit API keys to version control

## Testing and Validation

**ADK Web UI Testing**:
- Use `adk web` for interactive agent testing
- Monitor agent events and tool calls
- Validate state changes and tool outputs

**JSON Parsing Utilities**:
- Use `utils._parse_json_string()` for robust JSON parsing
- Handle malformed JSON gracefully with error reporting
- Maintain structured logging for debugging

## Video Content Integration

When working with video content analysis or YouTube metadata:
- Use structured metadata format (title, description, tags, supporting_docs)
- Implement multi-phase workflows: criticism → refinement → voting → aggregation
- Support multiple LLM providers for diverse perspectives
- Maintain audit trails of agent decisions and rankings