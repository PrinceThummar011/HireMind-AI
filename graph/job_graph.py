import os
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from agents.ats_agent import ATSAgent
from agents.cover_letter_agent import CoverLetterAgent
from agents.resume_agent import ResumeRewriterAgent
from agents.skill_gap_agent import SkillGapAgent


class JobAssistantState(TypedDict, total=False):
    resume_text: str
    job_description: str
    ats_result: dict[str, Any]
    skill_gap: dict[str, Any]
    rewritten_resume: str
    cover_letter: str


def _build_llm() -> ChatGroq | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model_name, temperature=0.2, api_key=api_key)


def _ats_node(state: JobAssistantState) -> JobAssistantState:
    ats_agent = ATSAgent()
    ats_result = ats_agent.run(
        resume_text=state.get("resume_text", ""),
        job_description=state.get("job_description", ""),
    )
    return {"ats_result": ats_result}


def _skill_gap_node(state: JobAssistantState) -> JobAssistantState:
    llm = _build_llm()
    skill_gap_agent = SkillGapAgent()

    ats_result = state.get("ats_result", {})
    missing_keywords = ats_result.get("missing_keywords", [])
    skill_gap = skill_gap_agent.run(
        missing_keywords=missing_keywords,
        job_description=state.get("job_description", ""),
        ats_result=ats_result,
        llm=llm,
    )
    return {"skill_gap": skill_gap}


def _resume_rewriter_node(state: JobAssistantState) -> JobAssistantState:
    llm = _build_llm()
    resume_agent = ResumeRewriterAgent()

    ats_result = state.get("ats_result", {})
    rewritten_resume = resume_agent.run(
        resume_text=state.get("resume_text", ""),
        job_description=state.get("job_description", ""),
        missing_keywords=ats_result.get("missing_keywords", []),
        llm=llm,
    )
    return {"rewritten_resume": rewritten_resume}


def _cover_letter_node(state: JobAssistantState) -> JobAssistantState:
    llm = _build_llm()
    cover_letter_agent = CoverLetterAgent()
    cover_letter = cover_letter_agent.run(
        resume_text=state.get("resume_text", ""),
        job_description=state.get("job_description", ""),
        llm=llm,
    )
    return {"cover_letter": cover_letter}


def build_job_graph():
    graph = StateGraph(JobAssistantState)

    graph.add_node("ats_scorer", _ats_node)
    graph.add_node("skill_gap", _skill_gap_node)
    graph.add_node("resume_rewriter", _resume_rewriter_node)
    graph.add_node("cover_letter", _cover_letter_node)

    graph.add_edge(START, "ats_scorer")
    graph.add_edge("ats_scorer", "skill_gap")
    graph.add_edge("skill_gap", "resume_rewriter")
    graph.add_edge("resume_rewriter", "cover_letter")
    graph.add_edge("cover_letter", END)

    return graph.compile()


def run_job_assistant(resume_text: str, job_description: str) -> JobAssistantState:
    app = build_job_graph()
    initial_state: JobAssistantState = {
        "resume_text": resume_text,
        "job_description": job_description,
    }
    return app.invoke(initial_state)
