import re
from collections import Counter
from typing import Any

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
    "using",
    "used",
    "use",
    "etc",
    "across",
    "into",
    "can",
    "ability",
    "strong",
    "excellent",
    "plus",
    "preferred",
    "required",
    "requirements",
    "qualification",
    "qualifications",
    "experience",
    "work",
    "years",
    "year",
    "team",
    "role",
    "position",
    "job",
}

LOW_SIGNAL_WORDS = {
    "about",
    "actionable",
    "amounts",
    "analytical",
    "analyze",
    "anomalies",
    "applied",
    "best",
    "build",
    "business",
    "clearly",
    "collaborate",
    "company",
    "complex",
    "contribute",
    "core",
    "cross-functionally",
    "customer",
    "datasets",
    "decision-making",
    "decision",
    "degree",
    "design",
    "develop",
    "drive",
    "effective",
    "effectively",
    "effectiveness",
    "engineers",
    "evaluate",
    "expertise",
    "extract",
    "features",
    "field",
    "findings",
    "hands-on",
    "identify",
    "implement",
    "including",
    "insights",
    "join",
    "junior",
    "large",
    "leverage",
    "like",
    "location",
    "looking",
    "managers",
    "master",
    "mentor",
    "new",
    "non-technical",
    "optimize",
    "patterns",
    "practices",
    "predictive",
    "presentations",
    "problems",
    "proven",
    "quantitative",
    "related",
    "remote",
    "responsibilities",
    "scalable",
    "senior",
    "solutions",
    "solve",
    "strategic",
    "systems",
    "through",
    "title",
    "trends",
    "vast",
}

SKILL_TAXONOMY: dict[str, set[str]] = {
    "programming": {
        "python",
        "sql",
        "scala",
        "java",
        "c++",
        "bash",
    },
    "machine_learning": {
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "time series",
        "feature engineering",
        "model deployment",
        "mlops",
        "llm",
        "generative ai",
    },
    "data_and_ml_tools": {
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "xgboost",
        "lightgbm",
        "spark",
        "databricks",
        "airflow",
        "dbt",
        "git",
    },
    "analytics_and_experimentation": {
        "statistics",
        "hypothesis testing",
        "a/b testing",
        "experimentation",
        "data visualization",
        "tableau",
        "power bi",
        "storytelling",
    },
    "data_engineering": {
        "data pipelines",
        "etl",
        "data warehousing",
        "snowflake",
        "bigquery",
        "redshift",
    },
    "cloud_and_platforms": {
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
    },
    "business_and_collaboration": {
        "communication",
        "stakeholder management",
        "problem solving",
        "product thinking",
    },
}

ALIASES: dict[str, str] = {
    "machine-learning": "machine learning",
    "ml": "machine learning",
    "deep-learning": "deep learning",
    "natural language processing": "nlp",
    "prompt engineering": "llm",
    "large language models": "llm",
    "genai": "generative ai",
    "generative-ai": "generative ai",
    "pytorch lightning": "pytorch",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "apache spark": "spark",
    "powerbi": "power bi",
    "ab testing": "a/b testing",
    "a b testing": "a/b testing",
    "hypothesis-testing": "hypothesis testing",
    "data pipeline": "data pipelines",
    "pipelines": "data pipelines",
    "warehousing": "data warehousing",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "k8s": "kubernetes",
}

REQUIRED_CUES = (
    "must",
    "required",
    "minimum",
    "need",
    "hands-on",
    "proficiency",
    "strong",
    "experience with",
)

PREFERRED_CUES = (
    "preferred",
    "nice to have",
    "bonus",
    "plus",
    "good to have",
)


class ATSResult(BaseModel):
    score: int = Field(ge=0, le=100)
    matched_keywords: list[str]
    missing_keywords: list[str]
    jd_keywords: list[str]
    resume_keywords: list[str]
    priority_missing_keywords: list[str] = Field(default_factory=list)
    keyword_analysis: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9+#-]", "", token)
    return token


def _normalize_phrase(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("_", " ")
    text = ALIASES.get(text, text)
    return text


def _phrase_pattern(phrase: str) -> str:
    escaped = re.escape(phrase)
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"(?<!\w){escaped}(?!\w)"


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    return len(re.findall(_phrase_pattern(phrase), text, flags=re.IGNORECASE))


def _canonical_keyword(keyword: str) -> str:
    normalized = _normalize_phrase(keyword)
    normalized = ALIASES.get(normalized, normalized)
    return normalized


def _all_skill_terms() -> set[str]:
    terms: set[str] = set()
    for values in SKILL_TAXONOMY.values():
        terms.update(values)
    terms.update(ALIASES.keys())
    return terms


def _category_for_keyword(keyword: str) -> str:
    for category, terms in SKILL_TAXONOMY.items():
        if keyword in terms:
            return category
    return "other"


def _extract_ngram_candidates(text: str, top_n: int = 60) -> list[str]:
    tokens = [
        tok
        for tok in re.findall(r"[A-Za-z0-9+#-]{2,}", text.lower())
        if tok not in STOP_WORDS and tok not in LOW_SIGNAL_WORDS and len(tok) > 2
    ]
    phrases = []
    for n in (2, 3):
        for i in range(0, max(len(tokens) - n + 1, 0)):
            phrase = " ".join(tokens[i : i + n])
            if any(part in STOP_WORDS for part in phrase.split()):
                continue
            phrases.append(phrase)

    counts = Counter(phrases)
    return [p for p, c in counts.most_common(top_n) if c >= 2]


def _extract_taxonomy_skills(text: str) -> set[str]:
    text = text.lower()
    hits: set[str] = set()
    for term in _all_skill_terms():
        canonical = _canonical_keyword(term)
        if re.search(_phrase_pattern(term), text, flags=re.IGNORECASE):
            hits.add(canonical)
    return hits


def extract_keywords(text: str, top_n: int = 60) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9+#-]{2,}", text.lower())
    tokens = [_normalize_token(token) for token in raw_tokens]
    tokens = [
        token
        for token in tokens
        if token
        and token not in STOP_WORDS
        and token not in LOW_SIGNAL_WORDS
        and not token.isdigit()
        and len(token) >= 3
    ]

    frequency = Counter(tokens)
    common_tokens = [_canonical_keyword(token) for token, _ in frequency.most_common(top_n)]
    taxonomy_hits = list(_extract_taxonomy_skills(text))
    ngram_hits = [_canonical_keyword(p) for p in _extract_ngram_candidates(text, top_n=top_n)]

    merged: list[str] = []
    seen: set[str] = set()
    for token in taxonomy_hits + common_tokens + ngram_hits:
        if (
            not token
            or token in STOP_WORDS
            or token in LOW_SIGNAL_WORDS
            or token.isdigit()
            or len(token) < 3
        ):
            continue
        if token not in seen:
            merged.append(token)
            seen.add(token)

    return merged[:top_n]


def _rank_keywords_for_jd(job_description: str, jd_keywords: list[str]) -> list[dict[str, Any]]:
    jd_text = job_description.lower()
    sentences = [s.strip().lower() for s in re.split(r"[\n.!?]+", jd_text) if s.strip()]

    ranked: list[dict[str, Any]] = []
    for keyword in jd_keywords:
        frequency = _count_phrase_occurrences(jd_text, keyword)
        if frequency <= 0:
            continue

        sentence_hits = [s for s in sentences if re.search(_phrase_pattern(keyword), s)]
        required_hits = sum(any(cue in s for cue in REQUIRED_CUES) for s in sentence_hits)
        preferred_hits = sum(any(cue in s for cue in PREFERRED_CUES) for s in sentence_hits)

        score = frequency + (required_hits * 3) + preferred_hits
        ranked.append(
            {
                "keyword": keyword,
                "category": _category_for_keyword(keyword),
                "frequency": frequency,
                "importance_score": score,
                "required_context_hits": required_hits,
                "preferred_context_hits": preferred_hits,
            }
        )

    ranked.sort(
        key=lambda item: (
            item["importance_score"],
            item["frequency"],
            len(item["keyword"]),
        ),
        reverse=True,
    )
    return ranked


def _select_core_keywords(ranked_jd_keywords: list[dict[str, Any]], limit: int = 60) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def _add(keyword: str) -> None:
        if keyword in seen:
            return
        selected.append(keyword)
        seen.add(keyword)

    for item in ranked_jd_keywords:
        keyword = item["keyword"]
        category = item["category"]
        frequency = item["frequency"]
        importance = item["importance_score"]
        required_hits = item["required_context_hits"]

        if keyword in LOW_SIGNAL_WORDS or len(keyword) < 3:
            continue

        # Always keep taxonomy-backed terms.
        if category != "other":
            _add(keyword)
            continue

        # Keep non-taxonomy terms only if they are clearly important.
        if required_hits > 0 or frequency >= 2 or importance >= 3:
            _add(keyword)

        if len(selected) >= limit:
            break

    return selected[:limit]


def _coverage_by_category(jd_set: set[str], resume_set: set[str]) -> dict[str, dict[str, Any]]:
    by_category: dict[str, dict[str, Any]] = {}
    for category, terms in SKILL_TAXONOMY.items():
        jd_terms = sorted([term for term in jd_set if term in terms])
        if not jd_terms:
            continue
        matched_terms = sorted([term for term in jd_terms if term in resume_set])
        missing_terms = sorted([term for term in jd_terms if term not in resume_set])
        coverage = int(round((len(matched_terms) / len(jd_terms)) * 100)) if jd_terms else 0

        by_category[category] = {
            "coverage": coverage,
            "jd_terms": jd_terms,
            "matched": matched_terms,
            "missing": missing_terms,
        }

    return by_category


def _build_recommendations(priority_missing_keywords: list[str], coverage: dict[str, dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []

    if priority_missing_keywords:
        recommendations.append(
            "Prioritize these high-impact missing keywords: "
            + ", ".join(priority_missing_keywords[:8])
            + "."
        )

    low_coverage_categories = [
        (category, details["coverage"])
        for category, details in coverage.items()
        if details.get("coverage", 0) < 50
    ]
    low_coverage_categories.sort(key=lambda item: item[1])

    if low_coverage_categories:
        weakest = ", ".join([f"{name} ({pct}%)" for name, pct in low_coverage_categories[:3]])
        recommendations.append(f"Improve weakest skill areas first: {weakest}.")

    recommendations.append(
        "Add missing keywords only where true, and prove each with measurable achievements in project or experience bullets."
    )
    return recommendations


def compute_ats_score(resume_text: str, job_description: str) -> ATSResult:
    jd_keywords = extract_keywords(job_description, top_n=140)
    resume_keywords = extract_keywords(resume_text, top_n=180)

    jd_set = {_canonical_keyword(k) for k in jd_keywords}
    resume_set = {_canonical_keyword(k) for k in resume_keywords}

    if not jd_set:
        return ATSResult(
            score=0,
            matched_keywords=[],
            missing_keywords=[],
            jd_keywords=[],
            resume_keywords=resume_keywords,
            priority_missing_keywords=[],
            keyword_analysis={},
            recommendations=[],
        )

    ranked_jd_keywords = _rank_keywords_for_jd(job_description=job_description, jd_keywords=sorted(jd_set))

    core_jd_keywords = _select_core_keywords(ranked_jd_keywords, limit=60)
    core_jd_set = set(core_jd_keywords)

    if not core_jd_set:
        core_jd_set = jd_set

    matched = sorted(core_jd_set & resume_set)
    missing = sorted(core_jd_set - resume_set)
    score = int(round((len(matched) / len(core_jd_set)) * 100))

    priority_missing_keywords = [
        item["keyword"]
        for item in ranked_jd_keywords
        if item["keyword"] in missing
    ][:15]

    coverage = _coverage_by_category(jd_set=core_jd_set, resume_set=resume_set)

    keyword_analysis = {
        "top_keywords": ranked_jd_keywords[:30],
        "coverage_by_category": coverage,
        "method": "hybrid-rule-based (taxonomy + phrase frequency + requirement context weighting)",
        "core_keywords": sorted(core_jd_set),
    }

    recommendations = _build_recommendations(
        priority_missing_keywords=priority_missing_keywords,
        coverage=coverage,
    )

    return ATSResult(
        score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        jd_keywords=sorted(core_jd_set),
        resume_keywords=sorted(resume_set),
        priority_missing_keywords=priority_missing_keywords,
        keyword_analysis=keyword_analysis,
        recommendations=recommendations,
    )


class ATSAgent:
    def run(self, resume_text: str, job_description: str) -> dict:
        result = compute_ats_score(resume_text=resume_text, job_description=job_description)
        return result.model_dump()
