import re
from collections import Counter

from pydantic import BaseModel, Field


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
    "will",
    "we",
    "our",
    "this",
    "those",
    "these",
}


class ATSResult(BaseModel):
    score: int = Field(ge=0, le=100)
    matched_keywords: list[str]
    missing_keywords: list[str]
    jd_keywords: list[str]
    resume_keywords: list[str]


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9+#.-]", "", token)
    return token


def extract_keywords(text: str, top_n: int = 60) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9+#.-]{2,}", text.lower())
    tokens = [_normalize_token(token) for token in raw_tokens]
    tokens = [token for token in tokens if token and token not in STOP_WORDS]

    frequency = Counter(tokens)
    common_tokens = [token for token, _ in frequency.most_common(top_n)]
    return common_tokens


def compute_ats_score(resume_text: str, job_description: str) -> ATSResult:
    jd_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text, top_n=120)

    jd_set = set(jd_keywords)
    resume_set = set(resume_keywords)

    if not jd_set:
        return ATSResult(
            score=0,
            matched_keywords=[],
            missing_keywords=[],
            jd_keywords=[],
            resume_keywords=resume_keywords,
        )

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)
    score = int(round((len(matched) / len(jd_set)) * 100))

    return ATSResult(
        score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        jd_keywords=sorted(jd_set),
        resume_keywords=sorted(resume_set),
    )


class ATSAgent:
    def run(self, resume_text: str, job_description: str) -> dict:
        result = compute_ats_score(resume_text=resume_text, job_description=job_description)
        return result.model_dump()
