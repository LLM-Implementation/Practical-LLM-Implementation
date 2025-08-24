# Practical-LLM-Implementation

Practical code examples and tutorials for fine-tuning Large Language Models and building AI agent systems. Companion repo for YouTube guides.

[![LLM Implementation](./images/LLM-Implementation-Channel.png)](www.youtube.com/@LLMImplementation)
Hi! Welcome to the companion repository for the **LLM Implementation** YouTube channel www.youtube.com/@LLMImplementation. Here, I share code, notebooks, and resources related to my learnings and practical experiences in LLM implementation, including tutorials and project explorations focused on **fine-tuning** and **AI agent development**.

This repository covers two main areas:
1. **Fine-tuning**: Adapt powerful pre-trained LLMs like Google Gemini, Llama, and GPT models to your specific tasks, domains, or desired output styles
2. **AI Agent Systems**: Build sophisticated multi-agent systems using frameworks like Google's Agent Development Kit (ADK), LangGraph, and AutoGen

## Repository Structure

### 🤖 AI Agent Systems

#### Google Agent Development Kit (ADK)
* **Folder:** [`adk/`](./adk/)
* **Description:** Multi-agent systems built with Google's ADK framework
* **Examples:**
    * **[Google Search Agent](./adk/google-search-agent/)**: Simple agent using Google Search tool with Gemini 2.5 Flash
    * **[YouTube Metadata Refinement Agent](./adk/youtube-metadata-refinement-agent/)**: Complex multi-LLM workflow with criticism → refinement → voting → aggregation phases
    * **[ADK Deploy Notebook](./adk/adk-deploy-notebook/)**: Deployment examples for ADK agents

#### Agent Frameworks
* **Folder:** [`agents_frameworks/`](./agents_frameworks/)
* **Description:** Examples using different agent frameworks
* **Examples:**
    * **LangGraph Basic**: Fundamental LangGraph implementations
    * **AutoGen (ag2) with OpenAI**: Multi-agent conversations and workflows

### 🔧 Fine-Tuning Examples

#### Google Gemini
* **Folder:** [`gemini/`](./gemini/)
* **Description:** Examples focusing on fine-tuning Google's Gemini family of models
* **Examples:**
    * **[Supervised Fine-Tuning to Humanize Prompts](./gemini/supervised_humanizing_prompts/)**: Data preparation (JSONL) and supervised tuning on Vertex AI *(Companion to: [Practical Gemini Fine-Tuning: Step-by-Step Guide with Vertex AI](https://youtu.be/MOaHlowhp8s))*
    * **[Gemini 2.5 Pro Video Analysis](./gemini/gemini_2_5_pro_video_analysis/)**: Video content analysis demonstrations
    * **[Prompt Rewriting with Gemini 2.5 Pro Exp](./gemini/gemini_2_5_Pro_Exp_Prompt_Rewriting/)**: Advanced prompt engineering techniques

#### Llama
* **Folder:** [`llama/`](./llama/)
* **Description:** Llama model fine-tuning examples
* **Examples:**
    * **Llama 3 Fine-tuning**: Practical fine-tuning implementations for Llama 3

#### GPT Open Source
* **Folder:** [`gpt-oss-20b/`](./gpt-oss-20b/)
* **Description:** Fine-tuning examples for open-source GPT models
* **Examples:**
    * **GPT OSS 20B Fine-tuning**: Large-scale model fine-tuning techniques

### 📝 Context Engineering

* **Folder:** [`context-engineering/`](./context-engineering/)
* **Description:** Framework for building AI features using context engineering principles
* **Key Concept:** 2-step process - setup examples, then use the universal magic prompt template


## Getting Started

### For ADK Agent Development

1. **Setup Environment**:
   ```bash
   cd adk/
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

2. **Install Dependencies**:
   ```bash
   pip install google-adk -q
   pip install python-dotenv -q
   pip install litellm -q
   ```

3. **Configure API Keys**:
   Create a `.env` file in the `adk/` directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

4. **Run ADK Web UI**:
   ```bash
   adk web
   ```
   Access at `http://localhost:8000`

### For Jupyter Notebooks

1. **Install Dependencies**:
   ```bash
   pip install langgraph langchain langsmith
   pip install jupyter notebook
   ```

2. **Launch Notebooks**:
   ```bash
   jupyter notebook
   ```

## Contributing

I'm sharing my learning journey here! If you find bugs, have suggestions for improvements, spot errors, or want to discuss ideas related to fine-tuning or agent development, feel free to open an issue. Pull requests that fix issues or add value are also welcome.

## License

This project is licensed under the [MIT License](./LICENSE). *(Assumes you chose the MIT License based on our discussion)*

---

Let's learn and explore LLM fine-tuning and AI agent development together! Don't forget to check out the www.youtube.com/@LLMImplementation for video walkthroughs and discussions.