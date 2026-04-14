from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage


class CoverLetterAgent:
    def run(
        self,
        resume_text: str,
        job_description: str,
        llm: BaseChatModel | None = None,
    ) -> str:
        if not llm:
            return (
                "Dear Hiring Manager,\n\n"
                "I am excited to apply for this role. My background aligns with your core requirements, "
                "and I am confident I can contribute quickly. I have attached a tailored resume and would "
                "welcome the chance to discuss how my skills match your team needs.\n\n"
                "Sincerely,\n"
                "Your Name"
            )

        prompt = f"""
Write a concise, tailored cover letter for the following candidate and job.

Rules:
- 220-320 words
- Professional and specific
- Highlight strongest relevant skills from the resume
- Include motivation for the role and company impact mindset
- No fake claims

Job Description:
{job_description}

Candidate Resume:
{resume_text}
""".strip()

        response = llm.invoke([HumanMessage(content=prompt)])
        return getattr(response, "content", str(response))
