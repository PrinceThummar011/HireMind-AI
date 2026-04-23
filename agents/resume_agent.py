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
You are an expert resume writer specializing in modern, high-impact professional resumes.
Rewrite the resume to align perfectly with the target job description.

Rules:
- **PRESERVE ALL SECTIONS**: Do NOT delete any sections from the original resume. Keep all the sections the user originally had (e.g., Summary, Experience, Education, Projects, Certifications).
- **ENHANCE, DO NOT RESTRUCTURE**: Your job is to improve the wording, grammar, and impact of the existing content. Do NOT completely restructure the resume or enforce a specific template if the user didn't have one.
- **ADD SKILLS IF NECESSARY**: If the job description requires certain skills that the user is missing, you may carefully add them to the "Skills" section (or create a "Skills" section if none exists), provided they fit the user's background.
- **ATS OPTIMIZATION**: Weave the provided keywords naturally into the existing bullet points and summary.
- **FORMATTING**: Use a clean, professional Markdown-style format (`###` for headers, `*` for bullets).
- **IMPACT**: Enhance existing bullet points with action verbs and quantify achievements where possible.
- **NO PLACEHOLDERS OR NOTES**: NEVER add meta-comments like "Note: Added skills" or "No experience provided". Just output the final, polished resume text.
- **TRUTHFULNESS**: Do NOT invent new job experiences, degrees, or companies.
- **PRESERVE PERSONAL DATA**: Keep name, email, phone, and links exactly as they are.

Job Description:
{job_description}

Keywords to emphasize if relevant:
{", ".join(missing_keywords[:20]) if missing_keywords else "None"}

Original Resume:
{resume_text}
""".strip()

        response = llm.invoke([HumanMessage(content=prompt)])
        return getattr(response, "content", str(response))
