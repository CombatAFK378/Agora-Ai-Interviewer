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
import threading
import time

import requests

from shared.config import get_settings

logger = logging.getLogger(__name__)

# Round-robin index across a pool of same-provider keys (e.g. several Groq keys
# from separate accounts, each with its own tokens-per-minute budget).
_key_lock = threading.Lock()
_key_idx = 0


def _next_key(keys: list[str]) -> str:
    global _key_idx
    if len(keys) == 1:
        return keys[0]
    with _key_lock:
        key = keys[_key_idx % len(keys)]
        _key_idx += 1
    return key

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
    use_fallback_chain: bool = True,
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
    chain = [primary]
    if use_fallback_chain:
        chain += [m for m in settings.fallback_models if m != primary]

    last_err: Exception | None = None
    for model_id in chain:
        provider, bare_model = _split(model_id)
        url, key_attr = PROVIDERS[provider]
        # Groq draws from a key pool (round-robin); other providers use one key.
        if provider == "groq":
            keys = settings.groq_keys
        else:
            k = getattr(settings, key_attr, "")
            keys = [k] if k else []
        if not keys:
            last_err = LLMError(f"no key for provider {provider!r} ({model_id})")
            logger.warning("skipping %s: no key configured", model_id)
            continue

        for attempt in range(retries + 1):
            api_key = _next_key(keys)   # rotate each attempt so a 429 retries on a fresh key
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


def see(
    prompt: str,
    image_url: str,
    *,
    model: str | None = None,
    max_tokens: int = 220,
    temperature: float = 0.3,
    timeout: float = 40.0,
    retries: int = 1,
) -> str:
    """Vision call (§8 coding round): assess an image. `image_url` is a data: URI
    (or URL). Tries a chain of VISION-capable models (text models can't see, so the
    normal text fallback chain is bypassed) — free vision tiers 429 often, so we
    fall through to the next one before giving up. Caller degrades gracefully."""
    settings = get_settings()
    primary = model or settings.llm_vision_model
    chain, seen = [], set()
    for m in [primary, *settings.vision_fallbacks]:   # dedupe, keep order
        if m and m not in seen:
            seen.add(m)
            chain.append(m)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]}]
    last: Exception | None = None
    for m in chain:
        try:
            return chat(messages, model=m, max_tokens=max_tokens, temperature=temperature,
                        timeout=timeout, retries=retries, use_fallback_chain=False)
        except Exception as e:
            last = e
            logger.warning("vision model %s failed (%s); trying next", m, e)
    raise LLMError(f"all vision models failed; last error: {last}")


def complete_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 400,
    timeout: float = 30.0,
    retries: int = 2,
) -> dict:
    """Like chat(), but offers `tools` and returns the assistant message dict
    (which may carry `tool_calls`). Single model + key rotation, no fallback
    chain — used for the Orchestrator's override tool (§7)."""
    settings = get_settings()
    model_id = model or settings.llm_fast_model
    provider, bare_model = _split(model_id)
    url, key_attr = PROVIDERS[provider]
    keys = settings.groq_keys if provider == "groq" else (
        [getattr(settings, key_attr, "")] if getattr(settings, key_attr, "") else [])
    if not keys:
        raise LLMError(f"no key for provider {provider!r}")

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        api_key = _next_key(keys)
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": bare_model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "tools": tools, "tool_choice": "auto"},
                timeout=timeout,
            )
            if resp.status_code in _RETRYABLE_STATUS:
                raise _Transient(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise _Transient(str(data["error"])[:120])
            return data["choices"][0]["message"]
        except _Transient as e:
            last_err = e
            time.sleep(min(0.5 * (2 ** attempt), 4.0))
        except Exception as e:
            last_err = e
            break
    raise LLMError(f"tool completion failed: {last_err}")


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
    # Some providers return HTTP 200 with an error object in the body (e.g. an
    # upstream capacity error). Treat capacity/5xx as transient so we retry.
    if isinstance(data, dict) and data.get("error"):
        err = data["error"] if isinstance(data["error"], dict) else {"message": data["error"]}
        msg = str(err.get("message", err))
        code = err.get("code")
        transient = code in (408, 425, 429, 500, 502, 503, 504) or any(
            k in msg.lower()
            for k in ("overload", "exhaust", "temporarily", "rate limit", "timeout", "capacity")
        )
        if transient:
            raise _Transient(f"upstream: {msg[:120]}")
        raise ValueError(f"LLM error: {msg[:160]}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"unexpected LLM response shape: {data}") from e
    if not content:  # some models return null/empty content on a refusal/error
        raise ValueError("empty content from model")
    return content.strip()
