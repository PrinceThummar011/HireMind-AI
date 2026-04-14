from fastmcp import FastMCP

from agents.ats_agent import compute_ats_score, extract_keywords


mcp = FastMCP(name="ai-job-assistant-tools")


@mcp.tool()
def keyword_analysis(text: str, top_n: int = 40) -> list[str]:
    """Extract top keywords from a given text."""
    return extract_keywords(text, top_n=top_n)


@mcp.tool()
def ats_score(resume_text: str, job_description: str) -> dict:
    """Compute ATS score + matched/missing keywords between resume and JD."""
    result = compute_ats_score(resume_text=resume_text, job_description=job_description)
    return result.model_dump()


def get_mcp_server() -> FastMCP:
    return mcp


if __name__ == "__main__":
    mcp.run()
