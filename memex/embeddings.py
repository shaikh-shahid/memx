import httpx


class OllamaEmbeddingError(Exception):
    pass


def embed_text(text: str, model: str, base_url: str, timeout: int) -> list[float]:
    url = f"{base_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        if response.status_code == 404:
            raise OllamaEmbeddingError(
                f"Model '{model}' not found in Ollama. Run: ollama pull {model}"
            )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaEmbeddingError(f"Ollama returned unexpected response: {data}")

        try:
            return [float(x) for x in embedding]
        except (TypeError, ValueError) as e:
            raise OllamaEmbeddingError("Ollama returned a non-numeric embedding vector.") from e

    except httpx.ConnectError as e:
        raise OllamaEmbeddingError(
            f"Cannot connect to Ollama at {base_url}. Is Ollama running? Try: ollama serve"
        ) from e
    except httpx.TimeoutException as e:
        raise OllamaEmbeddingError(
            f"Ollama timed out after {timeout}s. Try increasing timeout_seconds in ~/.memx/config.toml"
        ) from e
    except httpx.HTTPStatusError as e:
        raise OllamaEmbeddingError(
            f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise OllamaEmbeddingError(f"Ollama request failed: {e}") from e
