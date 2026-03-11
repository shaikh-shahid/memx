import difflib
import json
import re

import httpx

EXTRACTION_PROMPT = """You are a memory extraction assistant. Your job is to extract atomic, self-contained facts from the text below.

Rules:
- Each fact must make complete sense on its own, without any surrounding context
- Write facts as statements, not questions
- Be specific. \"Uses Supabase for auth\" is good. \"Uses a database\" is too vague.
- Do not summarize. Extract distinct facts.
- Ignore small talk, greetings, filler content
- Avoid generic statements like "Uses a database" or "Project is a web application"
- Maximum 15 facts per extraction
- If the text contains no meaningful facts worth remembering, return an empty array

Return ONLY a JSON array of strings. No explanation, no markdown, no preamble.

Example output:
[\"Prefers TypeScript over JavaScript\", \"Project is deployed on Railway\", \"Using Supabase for auth, not Clerk\"]

Text to extract from:
{text}"""

SUMMARY_PROMPT = """You are a concise technical summarizer.

Write a single standalone memory sentence (max 220 characters) that captures the most important takeaway from the text.
Rules:
- One sentence only
- Concrete and specific
- No bullet points
- No markdown

Text:
{text}
"""


class OllamaExtractionError(Exception):
    pass


GENERIC_FACT_PATTERNS = [
    re.compile(r"^uses a database\.?$", re.IGNORECASE),
    re.compile(r"^project is a (web )?application\.?$", re.IGNORECASE),
    re.compile(r"^project is a [a-z0-9_-]+ project\.?$", re.IGNORECASE),
    re.compile(r"^project has a (ui|lib|src) folder\.?$", re.IGNORECASE),
]


def normalize_fact(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s:%./+-]", "", value)
    value = value.replace("local-only", "local only")
    value = value.replace("zero ", "no ")
    value = re.sub(r"(\d+)\.(\d+)\s*m ops/s", r"\1m ops/s", value)
    return value


def is_low_signal_fact(text: str) -> bool:
    value = text.strip()
    if not value:
        return True
    if len(value) < 10:
        return True
    if len(value.split()) < 2:
        return True
    for pattern in GENERIC_FACT_PATTERNS:
        if pattern.match(value):
            return True
    return False


def dedupe_and_filter_facts(facts: list[str], similarity_threshold: float = 0.93) -> list[str]:
    kept: list[str] = []
    normalized_seen: list[str] = []

    for fact in facts:
        candidate = fact.strip()
        if is_low_signal_fact(candidate):
            continue
        normalized = normalize_fact(candidate)
        if normalized in normalized_seen:
            continue
        if any(
            difflib.SequenceMatcher(a=normalized, b=existing).ratio() >= similarity_threshold
            for existing in normalized_seen
        ):
            continue
        kept.append(candidate)
        normalized_seen.append(normalized)
    return kept


def _clean_json_blob(raw_text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    cleaned = cleaned.rstrip("`").strip()
    return cleaned


def extract_facts(text: str, model: str, base_url: str, timeout: int) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/generate"
    prompt = EXTRACTION_PROMPT.format(text=text.strip())

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    raw_text = ""
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        if response.status_code == 404:
            raise OllamaExtractionError(
                f"Model '{model}' not found in Ollama. Run: ollama pull {model}"
            )
        response.raise_for_status()
        data = response.json()
        raw_text = str(data.get("response", "")).strip()

        cleaned = _clean_json_blob(raw_text)
        facts = json.loads(cleaned)

        if not isinstance(facts, list):
            raise OllamaExtractionError(f"Expected a JSON array, got: {type(facts).__name__}")

        output = [f for f in facts if isinstance(f, str)]
        return dedupe_and_filter_facts(output)

    except json.JSONDecodeError as e:
        raise OllamaExtractionError(
            f"Could not parse extraction response as JSON: {e}. Raw: {raw_text[:200]}"
        ) from e
    except httpx.ConnectError as e:
        raise OllamaExtractionError(
            f"Cannot connect to Ollama at {base_url}. Is Ollama running? Try: ollama serve"
        ) from e
    except httpx.TimeoutException as e:
        raise OllamaExtractionError(f"Ollama timed out after {timeout}s during extraction.") from e
    except httpx.HTTPStatusError as e:
        raise OllamaExtractionError(
            f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise OllamaExtractionError(f"Ollama request failed: {e}") from e


def summarize_text(text: str, model: str, base_url: str, timeout: int) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": SUMMARY_PROMPT.format(text=text.strip()),
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        if response.status_code == 404:
            raise OllamaExtractionError(
                f"Model '{model}' not found in Ollama. Run: ollama pull {model}"
            )
        response.raise_for_status()
        data = response.json()
        summary = str(data.get("response", "")).strip()
        if not summary:
            raise OllamaExtractionError("Summary response was empty.")
        # Ensure single sentence-like line for storage stability.
        summary = " ".join(summary.splitlines()).strip()
        return summary[:220]
    except httpx.ConnectError as e:
        raise OllamaExtractionError(
            f"Cannot connect to Ollama at {base_url}. Is Ollama running? Try: ollama serve"
        ) from e
    except httpx.TimeoutException as e:
        raise OllamaExtractionError(f"Ollama timed out after {timeout}s during summary generation.") from e
    except httpx.HTTPStatusError as e:
        raise OllamaExtractionError(
            f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise OllamaExtractionError(f"Ollama request failed: {e}") from e
