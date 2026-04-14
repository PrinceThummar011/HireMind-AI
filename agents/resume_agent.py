from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage


class ResumeRewriterAgent:
    def run(
        self,
        resume_text: str,
        job_description: str,
        missing_keywords: list[str],
        llm: BaseChatModel | None = None,
    ) -> str:
        if not llm:
            return (
                "[LLM not configured] Suggested improvements:\n\n"
                "1) Add a strong summary aligned with the role.\n"
                f"2) Include missing keywords where true: {', '.join(missing_keywords[:15]) or 'None'}.\n"
                "3) Quantify impact in each experience bullet.\n"
                "4) Reorder sections to prioritize relevant skills and projects.\n\n"
                "Original resume text:\n"
                f"{resume_text}"
            )

        prompt = f"""
You are an expert resume writer.
Rewrite the resume to align with the target job description.

Rules:
- Keep it truthful: do not invent experience, degrees, or companies.
- Improve clarity, impact, and ATS keyword alignment.
- Preserve professional tone.
- Output in clean plain text format.

Job Description:
{job_description}

Keywords to emphasize if relevant:
{", ".join(missing_keywords[:20]) if missing_keywords else "None"}

Resume:
{resume_text}
""".strip()

        response = llm.invoke([HumanMessage(content=prompt)])
        return getattr(response, "content", str(response))
