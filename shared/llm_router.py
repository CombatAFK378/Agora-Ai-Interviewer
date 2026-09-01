"""The one place every LLM call goes through.

ARCHITECTURE §2 is explicit: "Every LLM call goes through one router module with
a fallback chain (primary -> secondary -> openrouter/free). Free model IDs rotate
without notice; never hardcode a single ID at a call site."

A call site names a *job* (fast vs reasoning) by passing the model id it wants,
and this module walks the fallback chain around it, retrying transient failures
(429 / 5xx) with exponential backoff and falling through to the next id on a hard
error.

Model ids may carry a provider prefix so one chain can span providers:

    groq:openai/gpt-oss-120b            -> Groq        (fast, sub-second)
    openrouter:nvidia/nemotron-...:free -> OpenRouter
    nvidia/nemotron-...:free            -> OpenRouter  (default, no prefix)

Both providers speak the OpenAI chat-completions shape, so the request/response
handling is identical; only the base URL and key differ. If a model's provider
has no key configured, that entry is skipped and the chain moves on — so the fast
path falls back to OpenRouter until a Groq key is set, then speeds up for free.

The shared Redis token bucket for global rate limiting (ARCHITECTURE §8) is a
scaling concern for many concurrent interviews and lands with the storage layer;
Phase 2 runs one interview, so per-call retry/backoff is enough here.
"""
import logging
import time

import requests

from shared.config import get_settings

logger = logging.getLogger(__name__)

# provider name -> (chat-completions URL, Settings attribute holding the key)
PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "openrouter_api_key"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "groq_api_key"),
}
DEFAULT_PROVIDER = "openrouter"

# Statuses worth retrying the *same* model on (transient); anything else means
# fall through to the next model in the chain.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised when every model in the chain has failed."""


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    timeout: float = 30.0,
    retries: int = 2,
    reasoning_effort: str | None = None,
) -> str:
    """Return the assistant message text for `messages`.

    `model` is the preferred id (defaults to the fast model); the configured
    fallback chain is appended after it. Raises LLMError if all fail.

    `reasoning_effort` ("low"/"medium"/"high") caps how much a reasoning model
    thinks before answering. gpt-oss reasons by default, which with a tight
    max_tokens can eat the whole budget and return empty content — "low" keeps
    the fast path fast and non-empty. It's only sent to gpt-oss models; other
    models use a different knob, so it's dropped for them rather than erroring.
    """
    settings = get_settings()
    primary = model or settings.llm_fast_model
    chain = [primary] + [m for m in settings.fallback_models if m != primary]

    last_err: Exception | None = None
    for model_id in chain:
        provider, bare_model = _split(model_id)
        url, key_attr = PROVIDERS[provider]
        api_key = getattr(settings, key_attr, "")
        if not api_key:
            last_err = LLMError(f"no key for provider {provider!r} ({model_id})")
            logger.warning("skipping %s: %s not set", model_id, key_attr.upper())
            continue

        for attempt in range(retries + 1):
            try:
                return _call(
                    url, api_key, bare_model, messages,
                    temperature, max_tokens, timeout, reasoning_effort,
                )
            except _Transient as e:
                last_err = e
                backoff = min(0.5 * (2 ** attempt), 4.0)
                logger.warning(
                    "LLM %s transient error (%s); retry %d/%d in %.1fs",
                    model_id, e, attempt + 1, retries, backoff,
                )
                time.sleep(backoff)
            except Exception as e:  # hard error — stop retrying this model
                last_err = e
                logger.warning("LLM %s failed (%s); trying next in chain", model_id, e)
                break

    raise LLMError(f"all models failed; last error: {last_err}")


def _split(model_id: str) -> tuple[str, str]:
    """('groq:openai/gpt-oss-120b') -> ('groq', 'openai/gpt-oss-120b').

    Splits only on a recognised provider prefix; otherwise the whole string is
    the model on the default provider (so 'nvidia/foo:free' stays intact).
    """
    if ":" in model_id:
        prefix, rest = model_id.split(":", 1)
        if prefix in PROVIDERS:
            return prefix, rest
    return DEFAULT_PROVIDER, model_id


class _Transient(Exception):
    """Internal marker for a retryable failure."""


def _call(
    url: str,
    api_key: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout: float,
    reasoning_effort: str | None = None,
) -> str:
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort and "gpt-oss" in model_id:
        body["reasoning_effort"] = reasoning_effort

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    if resp.status_code in _RETRYABLE_STATUS:
        raise _Transient(f"HTTP {resp.status_code}")
    resp.raise_for_status()

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"unexpected LLM response shape: {data}") from e
    if not content:  # some models return null/empty content on a refusal/error
        raise ValueError("empty content from model")
    return content.strip()
