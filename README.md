# AI Job Application Assistant

An AI-powered job application assistant that helps you analyze a resume against a job description, estimate ATS compatibility, identify skill gaps, and generate tailored application materials.

## Overview

This project combines Streamlit, Groq, LangChain, LangGraph, and FastMCP to create a multi-agent workflow for job application support. Upload a resume in PDF format, paste a job description, and receive a structured analysis with actionable outputs.

## Key Features

- Resume upload and PDF text extraction
- ATS match scoring with matched and missing keywords
- Skill gap analysis with a learning plan
- Tailored resume rewriting
- Cover letter generation
- Fast LLM responses using Groq

## Tech Stack

- UI: Streamlit
- Orchestration: LangGraph
- LLM Provider: Groq
- Prompting and chains: LangChain
- Tooling: FastMCP
- PDF parsing: PyMuPDF

## Project Structure

```text
AI Job Application Assistant/
├── app.py
├── main.py
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
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Prerequisites

- Python 3.10 or later
- A Groq API key
- A PDF resume file

## Installation

1. Clone the repository and open the project folder.
2. Install dependencies using [uv](https://docs.astral.sh/uv/).

Example:

```bash
# Sync dependencies from pyproject.toml
uv sync
```

Alternatively, if you are using the `requirements.txt`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root and add your Groq credentials:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=your_tavily_api_key_here
```

The model name is optional, but this is the default configured by the project.
`TAVILY_API_KEY` is optional and enables live web resource recommendations in the skill-gap learning plan.

## Usage

Run the Streamlit app:

```bash
uv run streamlit run app.py
```

Then:

1. Upload a resume PDF.
2. Paste the target job description.
3. Click Run AI Analysis.
4. Review the ATS score, keyword analysis, skill gaps, resume draft, and cover letter.

## Workflow

The LangGraph workflow is designed to:

1. Extract text from the uploaded resume.
2. Run ATS scoring and keyword analysis.
3. Identify missing skills and produce a learning plan.
4. Generate a tailored resume.
5. Generate a tailored cover letter.
6. Present all outputs in the Streamlit UI.

## MCP Tools

The `tools/mcp_tools.py` module exposes FastMCP tools used by the assistant, including:

- `keyword_analysis(text, top_n)`
- `ats_score(resume_text, job_description)`

## Deployment

To deploy on Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Connect the repository in Streamlit Community Cloud.
3. Add `GROQ_API_KEY` to Secrets.
4. Deploy the app.

## Notes

- `app.py` is the main Streamlit entrypoint.
- `main.py` is a lightweight helper that points to the Streamlit launch command.
- Keep your resume PDF text-based for best extraction results.

