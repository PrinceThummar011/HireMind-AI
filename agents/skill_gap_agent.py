import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

try:
    from langchain_community.tools import TavilySearchResults
except Exception:  # pragma: no cover
    TavilySearchResults = None


def _basic_learning_plan(missing_skills: list[str], coverage_by_category: dict | None = None) -> str:
    if not missing_skills:
        return "You already cover the major required skills for this job description."

    top = missing_skills[:10]
    lines = ["Prioritized learning plan (30 days):"]

    if coverage_by_category:
        lines.append("\nWeakest categories first:")
        ranked_categories = sorted(
            coverage_by_category.items(),
            key=lambda item: item[1].get("coverage", 0),
        )
        for category, details in ranked_categories[:3]:
            lines.append(f"- {category.replace('_', ' ').title()}: {details.get('coverage', 0)}% coverage")

    lines.append("\nTop missing skills to target:")
    for index, skill in enumerate(top, start=1):
        lines.append(f"{index}. Learn or strengthen: {skill}")

    lines.extend(
        [
            "\nWeekly action plan:",
            "Week 1: Cover core foundations + one mini-project.",
            "Week 2: Build one end-to-end portfolio project using top 2-3 missing skills.",
            "Week 3: Add measurable outcomes, write clean README, and publish code.",
            "Week 4: Update resume bullets and LinkedIn with new proof of skills.",
            "\nRule: add keywords only where you can prove them with real work.",
        ]
    )
    return "\n".join(lines)


def _format_resources_block(resources: list[dict]) -> str:
    if not resources:
        return "No web resources available. Add TAVILY_API_KEY to enable live resource recommendations."

    lines = ["\nRecommended resources (live web search):"]
    for index, resource in enumerate(resources[:10], start=1):
        title = resource.get("title", "Resource")
        url = resource.get("url", "")
        source = resource.get("source", "web")
        summary = resource.get("summary", "")
        line = f"{index}. {title} ({source})"
        if url:
            line += f" - {url}"
        if summary:
            line += f"\n   Why useful: {summary}"
        lines.append(line)
    return "\n".join(lines)


def _build_search_queries(missing_skills: list[str], job_description: str) -> list[str]:
    top_skills = missing_skills[:5]
    if not top_skills:
        return []

    jd_focus = "data science"
    jd_lower = (job_description or "").lower()
    if "machine learning" in jd_lower:
        jd_focus = "machine learning"
    elif "data analyst" in jd_lower:
        jd_focus = "data analytics"

    queries: list[str] = []
    for skill in top_skills:
        queries.append(f"best {skill} learning resources for {jd_focus} 2026")
        queries.append(f"{skill} official documentation tutorial")

    # Keep request size reasonable
    return queries[:8]


def _search_learning_resources(missing_skills: list[str], job_description: str) -> list[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    if TavilySearchResults is None:
        return []

    queries = _build_search_queries(missing_skills=missing_skills, job_description=job_description)
    if not queries:
        return []

    try:
        search_tool = TavilySearchResults(max_results=3, search_depth="basic")
    except Exception:
        return []

    merged: list[dict] = []
    seen_urls: set[str] = set()

    for query in queries:
        try:
            response = search_tool.invoke({"query": query})
        except Exception:
            continue

        # Tavily response can vary by integration version.
        candidates = response
        if isinstance(response, dict):
            candidates = response.get("results", [])

        if not isinstance(candidates, list):
            continue

        for item in candidates:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue

            title = (item.get("title") or "").strip() or "Resource"
            source = (item.get("source") or "web").strip() or "web"
            content = (item.get("content") or "").strip()
            summary = content[:180] + ("..." if len(content) > 180 else "")

            merged.append(
                {
                    "title": title,
                    "url": url,
                    "source": source,
                    "summary": summary,
                }
            )
            seen_urls.add(url)

    return merged[:12]


def _priority_missing_from_ats(
    missing_keywords: list[str],
    ats_result: dict | None,
) -> list[str]:
    if not ats_result:
        return missing_keywords

    priority = ats_result.get("priority_missing_keywords", [])
    if not priority:
        analysis = ats_result.get("keyword_analysis", {})
        top_keywords = analysis.get("top_keywords", [])
        missing_set = set(missing_keywords)
        priority = [item.get("keyword") for item in top_keywords if item.get("keyword") in missing_set]

    merged: list[str] = []
    seen: set[str] = set()
    for skill in priority + missing_keywords:
        if not skill:
            continue
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
    return merged


class SkillGapAgent:
    def run(
        self,
        missing_keywords: list[str],
        job_description: str,
        ats_result: dict | None = None,
        llm: BaseChatModel | None = None,
        use_web_search: bool = True,
    ) -> dict:
        prioritized_missing = _priority_missing_from_ats(
            missing_keywords=missing_keywords,
            ats_result=ats_result,
        )
        missing_skills = prioritized_missing[:20]
        coverage_by_category = (ats_result or {}).get("keyword_analysis", {}).get("coverage_by_category", {})
        resource_recommendations = (
            _search_learning_resources(missing_skills=missing_skills, job_description=job_description)
            if use_web_search
            else []
        )

        if not llm:
            learning_plan = _basic_learning_plan(
                missing_skills=missing_skills,
                coverage_by_category=coverage_by_category,
            )
            learning_plan = f"{learning_plan}\n{_format_resources_block(resource_recommendations)}"
            return {
                "missing_skills": missing_skills,
                "learning_plan": learning_plan,
                "priority_missing_skills": missing_skills[:10],
                "resource_recommendations": resource_recommendations,
            }

        prompt = f"""
You are a career coach.
Given the job description and missing skills, create a concise but high-impact learning plan.

Job Description:
{job_description}

Missing Skills:
{", ".join(missing_skills) if missing_skills else "None"}

Coverage by Category:
{coverage_by_category if coverage_by_category else "Not available"}

Web search resources (curated real links):
{resource_recommendations if resource_recommendations else "Not available"}

Return plain text with:
1) Top priority missing skills first
2) Practical resources/types (courses/docs/projects) and explicitly cite the provided web links
3) 30-day weekly action plan
4) Resume-proof plan: how to show each new skill with metrics/truthful evidence
""".strip()

        response = llm.invoke([HumanMessage(content=prompt)])
        learning_plan = getattr(response, "content", str(response))
        return {
            "missing_skills": missing_skills,
            "priority_missing_skills": missing_skills[:10],
            "learning_plan": learning_plan,
            "resource_recommendations": resource_recommendations,
        }
