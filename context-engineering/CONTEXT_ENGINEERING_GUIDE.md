---

# 🚀 The Ultimate Context Engineering Guide

*A Simple 2-Step Process to Build AI Features*

---

## **The Process**

```
1. Setup Examples → 2. Use the Magic Prompt → Done! 🎉
```

*You provide the patterns, the AI does the work.*

---

## **Step 1: Setup Examples (5-10 Minutes)**

Place **2-5 high-quality, relevant code examples** in your `examples/` folder.

This is the most important step. **Good examples are better than long instructions.**

---

## **Step 2: The Universal Magic Prompt**

This is the only prompt you need. Copy the template below and paste it into Cursor, filling in the `[BRACKETS]`.

---

## Goal

Set up the context engineering environment for a **\[FEATURE DESCRIPTION]** using **\[FRAMEWORK]**, following `/examples/[REFERENCE-EXAMPLE]` as a reference — especially focusing on its key tools and approaches.

---

## Docs & References

* My project/channel details are in `@context_youtube_channel.txt`
* The framework documentation is here: **\[LINK TO DOCS]**

---

## Project Files to Update

Please update the following based on this setup:

* `@CLAUDE.md` with relevant project rules and framework-specific guidelines
* `@INITIAL.md` with a clear feature request — following the context engineering workflow

---

## Special Instructions

* **Interaction Mode**: I prefer to interact via **\[INTERACTION METHOD]**.
* Keep everything as simple as possible for a demo.
* Stick closely to the examples provided and avoid overengineering.

---

## After Setup — Use These Commands:

```bash
# 1. Generate the implementation plan
/generate-prp INITIAL.md

# 2. Execute the plan to build the feature
/execute-prp PRPs/your-feature-name.md
```

---

## The Template Prompt

```
Please scan the project files. The goal is to set up the context engineering environment for a [FEATURE DESCRIPTION] using [FRAMEWORK] — following the /examples/[REFERENCE-EXAMPLE] as a reference, especially its use of [KEY TOOL].

My project/channel details are in @context_youtube_channel.txt, and the docs are here: [LINK TO DOCS].  

Please update @CLAUDE.md with relevant project rules and @INITIAL.md with a clear feature request, following the context engineering workflow.

Special Instructions:
- Interaction Mode: I prefer to interact via [INTERACTION METHOD].
- Keep everything as simple as possible for a demo.
- Stick closely to the examples provided and avoid overengineering.
```

---

## ✅ Completed Examples

**Example 1: Google ADK Agent (Your Original Project)**

```
Please scan the project files. The goal is to set up the context engineering environment for a demo content creator assistant using Google ADK — following the /examples/research-agent as a reference, especially its use of the Google Search tool.

My project/channel details are in @context_youtube_channel.txt, and the docs are here: https://google.github.io/adk-docs/.  

Please update @CLAUDE.md with relevant project rules and @INITIAL.md with a clear feature request, following the context engineering workflow.

Special Instructions:
- Interaction Mode: I prefer to interact via adk web.
- Keep everything as simple as possible for a demo.
- Stick closely to the examples provided and avoid overengineering.
```

**Example 2: LangChain RAG System**

```
Please scan the project files. The goal is to set up the context engineering environment for a PDF Question-Answering Bot using LangChain — following the /examples/rag-bot as a reference, especially its integration with FAISS.

My project/channel details are in @context_youtube_channel.txt, and the docs are here: https://python.langchain.com/docs/.  

Please update @CLAUDE.md with relevant project rules and @INITIAL.md with a clear feature request, following the context engineering workflow.

Special Instructions:
- Interaction Mode: I prefer to interact via Streamlit web app.
- Use FAISS for vector storage and ensure file upload accepts only .pdf files.
- Keep everything simple for a demo.
```

**Example 3: FastAPI Backend**

```
Please scan the project files. The goal is to set up the context engineering environment for a User Authentication API using FastAPI — following the /examples/auth-api as a reference.

My project/channel details are in @context_youtube_channel.txt, and the docs are here: https://fastapi.tiangolo.com/.  

Please update @CLAUDE.md with relevant project rules and @INITIAL.md with a clear feature request, following the context engineering workflow.

Special Instructions:
- Interaction Mode: I prefer to interact via REST API endpoints (e.g., Postman).
- Use Pydantic for request validation and Passlib for password handling.
- Keep the API simple and focused on authentication.
```

**Example 4: Data Science Dashboard**

```
Please scan the project files. The goal is to set up the context engineering environment for a Sales Analytics Dashboard using Streamlit — following the /examples/data-dashboard as a reference.

My project/channel details are in @context_youtube_channel.txt, and the docs are here: https://docs.streamlit.io/.  

Please update @CLAUDE.md with relevant project rules and @INITIAL.md with a clear feature request, following the context engineering workflow.

Special Instructions:
- Interaction Mode: I prefer to interact via a web dashboard.
- The dashboard should include a bar chart, a line chart, and a filterable data table.
- Use Plotly for all visualizations.
- Keep everything simple for a demo.
```

---