import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from graph.job_graph import run_job_assistant
from utils.pdf_parser import extract_text_from_uploaded_pdf


def _render_keyword_list(title: str, items: list[str], color: str) -> None:
    st.markdown(f"### {title}")
    if not items:
        st.info("No items found.")
        return

    chips = " ".join([f":{color}[`{item}`]" for item in items])
    st.markdown(chips)


def _render_results(results: dict[str, Any]) -> None:
    ats_result = results.get("ats_result", {})
    skill_gap = results.get("skill_gap", {})
    rewritten_resume = results.get("rewritten_resume", "")
    cover_letter = results.get("cover_letter", "")

    score = ats_result.get("score", 0)
    matched_keywords = ats_result.get("matched_keywords", [])
    missing_keywords = ats_result.get("missing_keywords", [])

    st.success("Analysis complete.")
    st.metric("ATS Match Score", f"{score}%")
    st.progress(min(max(score / 100, 0.0), 1.0))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Keyword Analysis", "Skill Gap", "Rewritten Resume", "Cover Letter"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            _render_keyword_list("Matched Keywords", matched_keywords, "green")
        with col2:
            _render_keyword_list("Missing Keywords", missing_keywords, "red")

    with tab2:
        st.markdown("### Missing Skills")
        for skill in skill_gap.get("missing_skills", []):
            st.write(f"- {skill}")

        st.markdown("### Learning Plan")
        st.write(skill_gap.get("learning_plan", "No learning plan generated."))

    with tab3:
        st.text_area("Tailored Resume", rewritten_resume, height=420)
        st.download_button(
            "Download Rewritten Resume",
            data=rewritten_resume,
            file_name="rewritten_resume.txt",
            mime="text/plain",
        )

    with tab4:
        st.text_area("Tailored Cover Letter", cover_letter, height=420)
        st.download_button(
            "Download Cover Letter",
            data=cover_letter,
            file_name="cover_letter.txt",
            mime="text/plain",
        )


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="AI Job Application Assistant", page_icon="🧠", layout="wide")
    st.title("🧠 AI Job Application Assistant")
    st.caption("Streamlit + Groq + LangChain + LangGraph + FastMCP")

    groq_key_exists = bool(os.getenv("GROQ_API_KEY"))
    with st.sidebar:
        st.header("System Status")
        st.write(f"Groq API Key: {'✅ Configured' if groq_key_exists else '⚠️ Missing'}")
        st.info("Add GROQ_API_KEY in .env (local) or Streamlit Cloud secrets.")

    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Job Description", height=260)

    run_clicked = st.button("Run AI Analysis", type="primary")

    if run_clicked:
        if not uploaded_resume:
            st.error("Please upload a resume PDF.")
            return
        if not job_description.strip():
            st.error("Please paste a job description.")
            return

        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_uploaded_pdf(uploaded_resume)

        if not resume_text.strip():
            st.error("Could not extract text from the uploaded PDF.")
            return

        with st.spinner("Running multi-agent workflow..."):
            try:
                results = run_job_assistant(resume_text=resume_text, job_description=job_description)
            except Exception as exc:
                st.exception(exc)
                return

        _render_results(results)


if __name__ == "__main__":
    main()
