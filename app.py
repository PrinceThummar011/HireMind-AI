import os
import logging
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from graph.job_graph import run_job_assistant
from utils.pdf_parser import extract_text_from_uploaded_pdf


logger = logging.getLogger(__name__)

MAX_RESUME_CHARS = 18_000
MAX_JOB_DESCRIPTION_CHARS = 12_000
CHUNK_SIZE = 4_000
MAX_CHUNKS = 5


def _truncate_with_chunks(
    text: str,
    *,
    max_chars: int,
    chunk_size: int = CHUNK_SIZE,
    max_chunks: int = MAX_CHUNKS,
) -> tuple[str, bool, int, int]:
    normalized = (text or "").strip()
    original_len = len(normalized)
    if original_len <= max_chars:
        return normalized, False, original_len, original_len

    chunks = [normalized[i : i + chunk_size] for i in range(0, original_len, chunk_size)]

    kept_chunks: list[str] = []
    used_chars = 0
    for chunk in chunks[:max_chunks]:
        remaining = max_chars - used_chars
        if remaining <= 0:
            break

        if len(chunk) <= remaining:
            kept_chunks.append(chunk)
            used_chars += len(chunk)
        else:
            kept_chunks.append(chunk[:remaining])
            used_chars += remaining
            break

    truncated_text = "\n\n".join(kept_chunks).strip()
    truncated_len = len(truncated_text)
    was_truncated = truncated_len < original_len
    return truncated_text, was_truncated, original_len, truncated_len


def _render_keyword_list(title: str, items: list[str], color: str) -> None:
    st.markdown(f"### {title}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    cleaned = cleaned[:24]

    if not cleaned:
        st.info("No items found.")
        return

    chips = " ".join([f":{color}[`{item}`]" for item in cleaned])
    st.markdown(chips)


def _render_advanced_keyword_analysis(ats_result: dict[str, Any]) -> None:
    priority_missing = ats_result.get("priority_missing_keywords", [])
    keyword_analysis = ats_result.get("keyword_analysis", {})
    coverage_by_category = keyword_analysis.get("coverage_by_category", {})
    top_keywords = keyword_analysis.get("top_keywords", [])
    recommendations = ats_result.get("recommendations", [])

    if priority_missing:
        st.markdown("### Priority Missing Keywords")
        st.warning(
            ", ".join(priority_missing[:12])
        )

    if coverage_by_category:
        st.markdown("### Coverage by Category")
        coverage_rows = []
        for category, details in coverage_by_category.items():
            coverage_rows.append(
                {
                    "category": category.replace("_", " ").title(),
                    "coverage_%": details.get("coverage", 0),
                    "matched": len(details.get("matched", [])),
                    "missing": len(details.get("missing", [])),
                }
            )
        coverage_rows = sorted(coverage_rows, key=lambda row: row["coverage_%"])
        st.dataframe(coverage_rows, use_container_width=True, hide_index=True)

    if top_keywords:
        st.markdown("### Top JD Keywords (Weighted)")
        top_rows = [
            {
                "keyword": item.get("keyword", ""),
                "category": item.get("category", "other"),
                "importance_score": item.get("importance_score", 0),
                "frequency": item.get("frequency", 0),
            }
            for item in top_keywords[:20]
        ]
        st.dataframe(top_rows, use_container_width=True, hide_index=True)

    if recommendations:
        st.markdown("### ATS Recommendations")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")


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

        _render_advanced_keyword_analysis(ats_result)

    with tab2:
        st.markdown("### Missing Skills")
        for skill in skill_gap.get("missing_skills", []):
            st.write(f"- {skill}")

        resources = skill_gap.get("resource_recommendations", [])
        st.markdown("### Recommended Resources")
        if resources:
            for item in resources[:10]:
                title = item.get("title", "Resource")
                url = item.get("url", "")
                source = item.get("source", "web")
                summary = item.get("summary", "")

                if url:
                    st.markdown(f"- [{title}]({url}) ({source})")
                else:
                    st.markdown(f"- {title} ({source})")

                if summary:
                    st.caption(summary)
        else:
            st.info("No live resources found. Add TAVILY_API_KEY to enable web-based learning resources.")

        st.markdown("### Learning Plan")
        st.write(skill_gap.get("learning_plan", "No learning plan generated."))

    with tab3:
        st.text_area("Tailored Resume", rewritten_resume, height=420)

    with tab4:
        st.text_area("Tailored Cover Letter", cover_letter, height=420)


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

    if "results" not in st.session_state:
        st.session_state.results = None

    run_clicked = st.button("Run AI Analysis", type="primary")

    if run_clicked:
        st.session_state.results = None
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

        safe_job_description, jd_truncated, jd_original_len, jd_truncated_len = _truncate_with_chunks(
            job_description,
            max_chars=MAX_JOB_DESCRIPTION_CHARS,
        )
        safe_resume_text, resume_truncated, resume_original_len, resume_truncated_len = _truncate_with_chunks(
            resume_text,
            max_chars=MAX_RESUME_CHARS,
        )

        if jd_truncated:
            st.warning(
                f"Job description is very large, so only the first {jd_truncated_len:,} of {jd_original_len:,} characters are used for AI analysis."
            )
        if resume_truncated:
            st.warning(
                f"Resume text is very large, so only the first {resume_truncated_len:,} of {resume_original_len:,} characters are used for AI analysis."
            )

        with st.spinner("Running multi-agent workflow..."):
            try:
                results = run_job_assistant(
                    resume_text=safe_resume_text,
                    job_description=safe_job_description,
                )
                st.session_state.results = results
            except Exception as exc:
                logger.exception("Job assistant workflow failed: %s", exc)
                st.error("Something went wrong while running the analysis. Please try again in a moment.")
                with st.expander("Technical details"):
                    st.text(str(exc))
                return

    if st.session_state.results:
        _render_results(st.session_state.results)


if __name__ == "__main__":
    main()
