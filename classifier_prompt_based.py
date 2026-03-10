"""
Unified LLM-based content annotation of online discourse.

Implements prompt-based classification from:
    Boukes et al. (2024) — Public Sphere benchmark
    See also: GLLM annotation bias analysis (prompt improvements)

All five metrics are applied per text in a single call.  Individual
metrics can be toggled on/off at initialisation.

Usage
-----
    >>> from metrics import PublicSphereMetrics
    >>> m = PublicSphereMetrics(
    ...     hf_token="hf_...",
    ...     model="meta-llama/Llama-3.1-70B-Instruct",
    ...     metrics=["rationality", "incivility"],
    ... )
    >>> m(["Some political comment", "lol no"])
    [
        {"metrics": {"rationality": "No", "incivility": "No"}},
        {"metrics": {"rationality": "No", "incivility": "Yes"}},
    ]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

# Default HF Inference Providers endpoint (replaces deprecated
# api-inference.huggingface.co since 2025).


# ---------------------------------------------------------------------------
# Prompt definitions
#
# F1 scores are macro-avg on the held-out TEST set (n=773) evaluated
# against human gold labels.  Source: notebook 26, cells 93
# (Llama 3.1:70b), 142 (GPT-4o), 164 (GPT-4 Turbo).
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {

    # Best test-set F1 (macro): Llama 0.64, GPT-4o 0.54, GPT-4T 0.55
    "rationality": {
        "template": (
            "Does this comment provide rational analysis?\n"
            "Instructions: Code Yes (1) if the comment includes:\n"
            "Context or background,\n"
            "Evidence (facts, sources, authorities),\n"
            "Reasoning or structured argument.\n"
            "Code No (0) if these are absent\n"
            "\\n\\nRespond with only the predicted class (0 or 1) "
            "of the request.\\n\\n"
            "Text: {text}\\nClass:"
        ),
        "classes": {"0": "No", "1": "Yes"},
    },

    # Best test-set F1 (macro): Llama 0.75, GPT-4o 0.77, GPT-4T 0.80
    "incivility": {
        "template": (
            "Does this comment display incivility?\n"
            "Instructions: Code Yes (1) if the comment includes "
            "name-calling, insults, inflammatory language, sarcasm, "
            "shouting (ALL CAPS), vulgarity, discrimination, threats, "
            "or restrictions on rights. "
            "Code No (0) if none of these are present.\n\n"
            "\\n\\nRespond with only the predicted class (0 or 1) "
            "of the request.\\n\\n"
            "Text: {text}\\nClass:"
        ),
        "classes": {"0": "No", "1": "Yes"},
    },

    # Best test-set F1 (macro): Llama 0.66, GPT-4o 0.62, GPT-4T 0.63
    "interactivity": {
        "template": (
            "Does this comment acknowledge or respond to another "
            "user's comment?\n"
            "Instructions: Code Yes (1) if the comment shows agreement "
            "or disagreement with a specific user's statement, often "
            "signaled by a username or phrases like 'Yes,' 'No,' or "
            "'I agree.' Code No (0) if it lacks a clear acknowledgment "
            "or is only an insult."
            "\\n\\nRespond with only the predicted class (0 or 1) "
            "of the request.\\n\\n"
            "Text: {text}\\nClass:"
        ),
        "classes": {"0": "No", "1": "Yes"},
    },

    # Best test-set F1 (macro, as binary dummies):
    #   LIBERAL:       Llama 0.77, GPT-4o 0.78, GPT-4T 0.74
    #   CONSERVATIVE:  Llama 0.81, GPT-4o 0.81, GPT-4T 0.75
    "political_ideology": {
        "template": (
            "Classify the following message as ideologically liberal (0), "
            "ideologically neutral (1), or ideologically conservative (2). "
            "Ideology here is defined in the context of the US political "
            "system. Messages with no ideological content are classified "
            "as neutral.\n\n"
            "Respond with only the predicted class (0 or 1 or 2) "
            "of the request.\n\n"
            "Text: {text}\nClass:"
        ),
        "classes": {"0": "liberal", "1": "neutral", "2": "conservative"},
    },

    # No gold-label evaluation in notebook 26; introduced in notebook 50
    # for the annotation bias study across models and datasets.
    "political_post": {
        "template": (
            "Classify the following message as following messages as "
            "political (1) or non-political (0). Political is defined "
            "as any message which is directly about a political topic, "
            "references political developments, or makes reference to "
            "a political figure, group, or agency. References to federal "
            "organisations are political, as are references to branches "
            "of government. Broad mentions of national economic "
            "developments are political, but discussions of individual "
            "stock prices are not.\n\n"
            "Respond with only the predicted class (0 or 1) "
            "of the request.\n\n"
            "Text: {text}\nClass:"
        ),
        "classes": {"0": "non-political", "1": "political"},
    },
}

ALL_METRICS = list(METRIC_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Unified classifier
# ---------------------------------------------------------------------------

@dataclass
class PublicSphereMetrics:
    """Classify texts on public-sphere dimensions via HF Inference Providers.

    Parameters
    ----------
    hf_token : str
        HuggingFace API token.
    model : str
        Model identifier on HuggingFace Hub
        (e.g. ``"meta-llama/Llama-3.1-70B-Instruct"``).
    base_url : str
        API base URL.  Defaults to the current HF router endpoint.
    metrics : list[str] | None
        Subset of metrics to activate.  Defaults to all five:
        ``rationality``, ``incivility``, ``interactivity``,
        ``political_ideology``, ``political_post``.
    temperature : float
        Sampling temperature (default 0 for reproducibility).
    seed : int
        Random seed passed to the endpoint.
    max_tokens : int
        Maximum tokens the model may generate per prompt.
    max_retries : int
        Retry count on transient HTTP errors.
    retry_wait : float
        Seconds between retries.
    """
    DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
    hf_token: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    metrics: Optional[List[str]] = None
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 15
    max_retries: int = 5
    retry_wait: float = 20.0

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = list(ALL_METRICS)
        unknown = set(self.metrics) - set(ALL_METRICS)
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. "
                f"Choose from {ALL_METRICS}."
            )

    # -- Public API ---------------------------------------------------------

    def __call__(
        self, texts: Sequence[str]
    ) -> List[Dict[str, Dict[str, str]]]:
        """Annotate every text on all active metrics.

        Returns
        -------
        list[dict]
            ``[{"metrics": {"rationality": "Yes", ...}}, ...]``
        """
        return [self._classify_one(t) for t in texts]

    # -- Internals ----------------------------------------------------------

    def _classify_one(self, text: str) -> Dict[str, Dict[str, str]]:
        result: Dict[str, str] = {}
        for name in self.metrics:
            spec = METRIC_REGISTRY[name]
            prompt = spec["template"].format(text=text)
            raw = self._query(prompt)
            result[name] = spec["classes"].get(raw)
        return {"metrics": result}

    def _query(self, prompt: str) -> str:
        """Send an OpenAI-compatible chat completion request."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=120,
                )
                if resp.status_code == 200:
                    return self._parse(resp.json())
                if resp.status_code in (429, 500, 503):
                    logger.warning(
                        "Attempt %d/%d — HTTP %d, retrying in %.0fs",
                        attempt, self.max_retries,
                        resp.status_code, self.retry_wait,
                    )
                    time.sleep(self.retry_wait)
                    continue
                logger.error("HTTP %d: %s", resp.status_code, resp.text)
                return ""
            except requests.RequestException as exc:
                logger.warning("Request failed: %s — retrying", exc)
                time.sleep(self.retry_wait)

        return ""

    @staticmethod
    def _parse(data: dict) -> str:
        """Extract content from an OpenAI-compatible chat response."""
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return ""