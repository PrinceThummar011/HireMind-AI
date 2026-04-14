from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage


def _basic_learning_plan(missing_skills: list[str]) -> str:
    if not missing_skills:
        return "You already cover the major required skills for this job description."

    top = missing_skills[:8]
    lines = ["Prioritized learning plan:"]
    for index, skill in enumerate(top, start=1):
        lines.append(f"{index}. Learn or strengthen: {skill}")
    lines.append("Build at least 1 project that demonstrates each high-priority skill.")
    return "\n".join(lines)


class SkillGapAgent:
    def run(
        self,
        missing_keywords: list[str],
        job_description: str,
        llm: BaseChatModel | None = None,
    ) -> dict:
        missing_skills = missing_keywords[:20]

        if not llm:
            return {
                "missing_skills": missing_skills,
                "learning_plan": _basic_learning_plan(missing_skills),
            }

        prompt = f"""
You are a career coach.
Given the job description and missing skills, create a concise learning plan.

Job Description:
{job_description}

Missing Skills:
{", ".join(missing_skills) if missing_skills else "None"}

Return plain text with:
1) Priority list
2) Practical resources/types (courses/docs/projects)
3) 30-day action plan
""".strip()

        response = llm.invoke([HumanMessage(content=prompt)])
        learning_plan = getattr(response, "content", str(response))
        return {"missing_skills": missing_skills, "learning_plan": learning_plan}
