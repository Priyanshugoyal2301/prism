"""
llm.py — Provider-agnostic LLM client.

Supports:
  - OpenAI-compatible REST APIs (OpenAI, Groq, Mistral, Ollama, etc.)
  - Google Gemini via AI Studio (LLM_PROVIDER=gemini)
  - Google Gemini via OpenAI-compatible shim (LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai)

If LLM_API_KEY is empty, the client raises RulesOnlyModeError so every caller
can gracefully fall back to its rule-based path. This ensures G1 works with
no LLM key.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import get_settings
from app.telemetry import get_telemetry
from app.schemas import TelemetryEventType


class RulesOnlyModeError(RuntimeError):
    """Raised when no LLM API key is configured — callers must use rule-based fallback."""
    pass


class LLMError(RuntimeError):
    """Wraps LLM API errors for structured handling."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class LLMResponse:
    def __init__(self, content: str, tokens_in: int, tokens_out: int, model: str):
        self.content = content
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.cost_usd = _estimate_cost(model, tokens_in, tokens_out)


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """
    Rough cost estimate in USD. Uses known pricing as of Sep 2026.
    Returns 0.0 for unknown models — never fabricated.
    """
    PRICING: dict[str, tuple[float, float]] = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.5-flash": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.0-pro": (0.50, 1.50),
        "mistral-large": (2.00, 6.00),
        "mistral-small": (0.20, 0.60),
    }
    for prefix, (in_price, out_price) in PRICING.items():
        if model.startswith(prefix):
            return (tokens_in * in_price + tokens_out * out_price) / 1_000_000
    return 0.0


# ── Gemini native API client ─────────────────────────────────────────────────

class GeminiNativeClient:
    """
    Calls the Gemini generateContent REST API directly.
    Used when LLM_PROVIDER=gemini and the base URL is the AI Studio endpoint.
    Converts OpenAI-style messages to Gemini contents format.
    """

    GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    def _messages_to_contents(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI messages format to Gemini contents format."""
        contents = []
        system_text = None
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_text = text  # handled separately via systemInstruction
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})
        return contents, system_text

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        """Returns (content, tokens_in, tokens_out)."""
        contents, system_text = self._messages_to_contents(messages)

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{self.GEMINI_BASE}/models/{self._model}:generateContent?key={self._api_key}"
        resp = await self._client.post(url, json=body)
        if resp.status_code != 200:
            raise LLMError(
                f"Gemini API returned {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )

        data = resp.json()
        try:
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from e

        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0)
        tokens_out = usage.get("candidatesTokenCount", 0)
        return text_out, tokens_in, tokens_out

    async def stream_generate(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Streaming via Gemini streamGenerateContent."""
        contents, system_text = self._messages_to_contents(messages)

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}

        url = (
            f"{self.GEMINI_BASE}/models/{self._model}:streamGenerateContent"
            f"?key={self._api_key}&alt=sse"
        )
        async with self._client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                content = await resp.aread()
                raise LLMError(f"Gemini stream error {resp.status_code}: {content[:200]}")
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                        delta = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def close(self) -> None:
        await self._client.aclose()


# ── Main LLM client (façade) ─────────────────────────────────────────────────

class LLMClient:
    """
    Provider-agnostic async LLM client.

    Provider selection (via LLM_PROVIDER env var):
      - "openai" (default): OpenAI-compatible REST API with Bearer auth
      - "gemini": Google Gemini AI Studio native API

    Gemini quick-start:
        LLM_PROVIDER=gemini
        LLM_API_KEY=<your-gemini-api-key>
        LLM_CONTROLLER_MODEL=gemini-2.0-flash
        LLM_SYNTHESIZER_MODEL=gemini-2.5-flash
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg

        if not cfg.llm_api_key.strip():
            self._openai_client: Optional[httpx.AsyncClient] = None
            self._gemini: Optional[GeminiNativeClient] = None
            return

        provider = cfg.llm_provider.lower()

        if provider == "gemini":
            # Use native Gemini client; model resolved per-call
            self._openai_client = None
            self._gemini = GeminiNativeClient(
                api_key=cfg.llm_api_key,
                model=cfg.llm_synthesizer_model,
                timeout_s=cfg.llm_timeout_s,
            )
        else:
            # OpenAI-compatible (default)
            self._openai_client = httpx.AsyncClient(
                base_url=cfg.llm_base_url,
                headers={
                    "Authorization": f"Bearer {cfg.llm_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(cfg.llm_timeout_s),
            )
            self._gemini = None

    @property
    def _is_gemini(self) -> bool:
        return self._gemini is not None

    def _resolve_model(self, model_alias: str) -> str:
        if model_alias == "fast":
            return self._cfg.llm_controller_model
        if model_alias == "strong":
            return self._cfg.llm_synthesizer_model
        return model_alias

    async def complete(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        model: str = "fast",
        json_mode: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Non-streaming completion. Raises RulesOnlyModeError if no key is set.
        """
        if self._openai_client is None and self._gemini is None:
            raise RulesOnlyModeError("No LLM_API_KEY configured. Running in rules-only mode.")

        resolved_model = self._resolve_model(model)
        tel = get_telemetry()
        t0 = time.perf_counter()

        last_error: Optional[Exception] = None
        for attempt in range(1, self._cfg.llm_max_retries + 2):
            try:
                if self._is_gemini:
                    # Swap model on GeminiNativeClient per call
                    self._gemini._model = resolved_model  # type: ignore[union-attr]
                    content, tokens_in, tokens_out = await self._gemini.generate(  # type: ignore[union-attr]
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                else:
                    body: dict[str, Any] = {
                        "model": resolved_model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if json_mode:
                        body["response_format"] = {"type": "json_object"}
                    resp = await self._openai_client.post("/chat/completions", json=body)  # type: ignore[union-attr]
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code != 200:
                        raise LLMError(
                            f"LLM API returned {resp.status_code}: {resp.text[:200]}",
                            status_code=resp.status_code,
                        )
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)

                elapsed_ms = (time.perf_counter() - t0) * 1000
                llm_resp = LLMResponse(content, tokens_in, tokens_out, resolved_model)
                await tel.emit(
                    TelemetryEventType.SYNTHESIS_DONE,
                    session_id,
                    model=resolved_model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=llm_resp.cost_usd,
                    latency_ms=round(elapsed_ms, 2),
                    attempt=attempt,
                )
                return llm_resp

            except RulesOnlyModeError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                if attempt <= self._cfg.llm_max_retries:
                    await asyncio.sleep(0.5 * attempt)
                continue
            except LLMError as e:
                if e.status_code in (429, 500, 502, 503) and attempt <= self._cfg.llm_max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    last_error = e
                    continue
                raise

        raise LLMError(f"LLM call failed after {attempt} attempts: {last_error}")

    async def complete_json(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        model: str = "fast",
        max_tokens: int = 1024,
        schema_name: str = "response",
    ) -> dict[str, Any]:
        """Wrapper that always returns parsed JSON. Retries once on parse failure."""
        for parse_attempt in range(2):
            resp = await self.complete(
                session_id=session_id,
                messages=messages,
                model=model,
                json_mode=True,
                max_tokens=max_tokens,
            )
            try:
                return json.loads(resp.content)
            except json.JSONDecodeError:
                if parse_attempt == 0:
                    messages = messages + [{
                        "role": "user",
                        "content": "Your previous response was not valid JSON. Reply with ONLY valid JSON, no markdown.",
                    }]
                    continue
                raise LLMError(f"LLM returned invalid JSON after retry: {resp.content[:200]}")
        raise LLMError("complete_json: exhausted retries")

    async def stream_complete(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        model: str = "strong",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields text tokens as they arrive."""
        if self._openai_client is None and self._gemini is None:
            raise RulesOnlyModeError("No LLM_API_KEY configured.")

        resolved_model = self._resolve_model(model)

        if self._is_gemini:
            self._gemini._model = resolved_model  # type: ignore[union-attr]
            async for token in self._gemini.stream_generate(  # type: ignore[union-attr]
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield token
        else:
            body = {
                "model": resolved_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            async with self._openai_client.stream("POST", "/chat/completions", json=body) as resp:  # type: ignore[union-attr]
                if resp.status_code != 200:
                    content = await resp.aread()
                    raise LLMError(f"Stream error {resp.status_code}: {content[:200]}")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def close(self) -> None:
        if self._openai_client:
            await self._openai_client.aclose()
        if self._gemini:
            await self._gemini.close()


# ── Module-level singleton ──────────────────────────────────────────────────
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
