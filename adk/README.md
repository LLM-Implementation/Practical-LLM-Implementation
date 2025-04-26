# YouTube Content Metadata Agent

A multi-LLM agent system for refining YouTube video metadata using ADK (Agent Development Kit).

## Getting Started

1.  **Setup Virtual Environment (Recommended):**
    Open your terminal in the project's root directory (`Practical-LLM-Implementation/adk`) and run:
    ```bash
    # Create the virtual environment
    python3 -m venv .venv

    # Activate the virtual environment
    # On macOS/Linux:
    source .venv/bin/activate
    ```


2.  **Install Dependencies:**
    With your virtual environment activated, run:
    ```bash
    pip install google-adk -q
    pip install python-dotenv -q
    pip install litellm -q
    ```

3.  **Configure API Keys:**
    Create a file named `.env` in the [`adk`](adk ) directory (alongside the `youtube-content-agent/` folder). Add your necessary API keys to this file, for example:
    ```dotenv
    # In adk/.env
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
    ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY_HERE
    ```
    *(Replace the placeholder values with your actual keys)*

4.  **Run the Agent Web UI:**
    Navigate to the [`adk`](adk ) directory in your terminal (if you aren't already there) and ensure your virtual environment is active. Then run:
    ```bash
    adk web
    ```
    This will start the ADK development web UI, typically accessible at `http://localhost:8000`. You should be able to select and interact with the agents defined in [`adk/youtube-metadata-refinement-agent/agent.py`](adk/youtube-metadata-refinement-agent/agent.py ).