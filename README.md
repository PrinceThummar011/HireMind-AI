# 🧠 AI Job Application Assistant

Professional multi-agent job application assistant built with Streamlit, Groq, LangChain, LangGraph, and FastMCP.

## ✨ Features

- 📄 Resume upload (PDF)
- 📋 Job description input
- 🎯 ATS score (match percentage)
- ✍️ Resume rewriting tailored to JD
- 💌 Cover letter generation
- 🔍 Skill gap analysis with learning plan
- 📊 Keyword analysis (matched vs missing)
- ⚡ Fast LLM responses using Groq

## 🏗️ Tech Stack

- UI: Streamlit
- LLM: Groq (`llama-3.3-70b-versatile` by default)
- Orchestration: LangGraph
- Prompting/chains: LangChain
- Tool server: FastMCP
- PDF parsing: PyMuPDF

## 📁 Project Structure

ai-job-assistant/
├── app.py
├── graph/
│   └── job_graph.py
├── agents/
│   ├── ats_agent.py
│   ├── resume_agent.py
│   ├── cover_letter_agent.py
│   └── skill_gap_agent.py
├── tools/
│   └── mcp_tools.py
├── utils/
│   └── pdf_parser.py
├── .env
├── requirements.txt
└── README.md

## 🚀 Local Setup

1. Create and activate a virtual environment
2. Install dependencies
3. Add your Groq API key
4. Run Streamlit

Example:

- Install dependencies from `requirements.txt`
- Set `GROQ_API_KEY` in `.env`
- Start app with Streamlit using `app.py`

## 🔐 Environment Variables

Create `.env` in project root:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

## 🔄 LangGraph Flow

1. Parse resume PDF
2. ATS scorer agent
3. Skill gap agent
4. Resume rewriter agent
5. Cover letter agent
6. Display all outputs in Streamlit

## 🌐 Streamlit Cloud Deployment

1. Push project to GitHub
2. Open Streamlit Community Cloud
3. Connect repository
4. Add `GROQ_API_KEY` in Secrets
5. Deploy

## 🧩 MCP Tools

`tools/mcp_tools.py` includes FastMCP tools:

- `keyword_analysis(text, top_n)`
- `ats_score(resume_text, job_description)`

Run MCP server from project root by launching the module script.

