"""
Ollama LLM client — wraps the Ollama REST API for text generation.

Provides:
    - OllamaClient : production client calling Ollama at localhost:11434
    - MockLLMClient : deterministic mock for testing without a running server

Usage:
    >>> from src.pipelines.shared.llm_client import OllamaClient
    >>> client = OllamaClient(model="mistral")
    >>> answer = client.generate("What is 2+2?")

    >>> # Auto-detect: use Ollama if available, else fall back to mock
    >>> client = create_llm_client(config)
"""

import time
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


# ────────────────────────────────────────────────
#  Ollama client
# ────────────────────────────────────────────────

class OllamaClient:
    """
    Low-level client for the Ollama REST API.

    Parameters
    ----------
    model : str
        Model name (e.g. 'mistral', 'llama3').
    base_url : str
        Ollama server URL.
    temperature : float
        Sampling temperature (0.0 = deterministic).
    max_tokens : int
        Maximum tokens to generate.
    timeout : int
        HTTP request timeout in seconds.
    seed : int or None
        Random seed for reproducibility (supported by recent Ollama versions).
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 300,
        seed: Optional[int] = 42,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed

        self._available: Optional[bool] = None  # cached availability

        logger.debug(
            f"OllamaClient initialised: model={model}, "
            f"url={self.base_url}, temp={temperature}"
        )

    # ── Availability check ───────────────────────

    def is_available(self) -> bool:
        """
        Check if the Ollama server is reachable and the model is loaded.

        Results are cached after the first successful check.
        """
        if self._available is not None:
            return self._available

        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if resp.status_code == 200:
                # Check if our model is in the list
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Ollama model names can include tags like "mistral:latest"
                available = any(
                    self.model in name for name in model_names
                )
                if not available:
                    logger.warning(
                        f"Ollama is running but model '{self.model}' not found. "
                        f"Available: {model_names}. Run: ollama pull {self.model}"
                    )
                self._available = available
                return available
        except Exception:
            pass

        self._available = False
        logger.warning(
            f"Ollama server not reachable at {self.base_url}. "
            f"Start with: ollama serve"
        )
        return False

    # ── Single generation ────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Generate a response for a single prompt.

        Parameters
        ----------
        prompt : str
            The full prompt string.

        Returns
        -------
        str
            The generated text response.

        Raises
        ------
        ConnectionError
            If Ollama is not reachable.
        RuntimeError
            If the API returns a non-200 status.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if self.seed is not None:
            payload["options"]["seed"] = self.seed

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is the server running? Start with: ollama serve"
            )
        except requests.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Try increasing the timeout or using a smaller model."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama API error (status {resp.status_code}): "
                f"{resp.text[:500]}"
            )

        result = resp.json()
        response_text = result.get("response", "").strip()

        logger.debug(
            f"Generated {len(response_text)} chars "
            f"(eval_duration={result.get('eval_duration', 'N/A')})"
        )
        return response_text

    # ── Batch generation ─────────────────────────

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts sequentially.

        Ollama does not support true batching, so prompts are processed
        one at a time. Use this for evaluation runs.

        Parameters
        ----------
        prompts : list[str]
            List of prompt strings.

        Returns
        -------
        list[str]
            Responses in the same order as prompts.
        """
        responses = []
        for i, prompt in enumerate(prompts):
            try:
                resp = self.generate(prompt)
            except (ConnectionError, TimeoutError, RuntimeError) as e:
                logger.error(f"Generation failed for prompt {i}: {e}")
                resp = ""
            responses.append(resp)

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{len(prompts)} responses")

        logger.info(f"Batch generation complete: {len(responses)} responses")
        return responses

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "OllamaClient":
        """
        Create client from merged pipeline config.

        Reads from config["llm"] and config["generation"].
        """
        llm = config.get("llm", {})
        gen = config.get("generation", {})

        return cls(
            model=llm.get("model", "mistral"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 512),
            timeout=gen.get("timeout_seconds", 120),
            seed=config.get("experiment", {}).get("seed", 42),
        )

    def __repr__(self) -> str:
        return (
            f"OllamaClient(model={self.model!r}, "
            f"url={self.base_url!r})"
        )


# ────────────────────────────────────────────────
#  Mock client (for testing)
# ────────────────────────────────────────────────

class MockLLMClient:
    """
    Deterministic mock LLM client for testing without Ollama.

    Returns a fixed response or echoes part of the prompt.

    Parameters
    ----------
    default_response : str
        The response to return for all prompts.
    model : str
        Model name (for logging/metadata).
    """

    def __init__(
        self,
        default_response: str = "[mock response]",
        model: str = "mock-model",
    ):
        self.default_response = default_response
        self.model = model
        self._call_count = 0
        self._prompts: List[str] = []

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        self._call_count += 1
        self._prompts.append(prompt)
        logger.debug(f"MockLLMClient.generate called (call #{self._call_count})")
        return self.default_response

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_prompt(self) -> Optional[str]:
        return self._prompts[-1] if self._prompts else None

    def __repr__(self) -> str:
        return f"MockLLMClient(model={self.model!r})"


# ────────────────────────────────────────────────
#  Factory function
# ────────────────────────────────────────────────

def create_llm_client(
    config: dict,
    fallback_to_mock: bool = True,
) -> "OllamaClient | MockLLMClient":
    """
    Create the appropriate LLM client based on availability.

    If Ollama is reachable, returns an OllamaClient.
    Otherwise, falls back to MockLLMClient (if allowed).

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    fallback_to_mock : bool
        If True, use MockLLMClient when Ollama is unavailable.
        If False, raise ConnectionError.
    """
    # Explicit parameter takes precedence; config is only a default
    gen = config.get("generation", {})
    fallback = fallback_to_mock if fallback_to_mock else gen.get("fallback_to_mock", False)

    client = OllamaClient.from_config(config)

    if client.is_available():
        logger.info(f"Using Ollama LLM: {client.model}")
        return client

    if fallback:
        logger.warning(
            f"Ollama unavailable — using MockLLMClient. "
            f"Answers will be placeholder text."
        )
        return MockLLMClient(model=client.model)

    raise ConnectionError(
        f"Ollama is not available at {client.base_url} and "
        f"fallback_to_mock is disabled."
    )
