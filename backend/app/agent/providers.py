"""
LLM provider abstraction and the local Ollama implementation.

The orchestrator depends only on ``ChatProvider``; another serving backend
can be added by implementing ``chat`` without touching orchestration code.
"""
from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
# Context window requested from Ollama, in tokens. Sent explicitly because
# when a prompt exceeds the server's default window (often 4096 tokens),
# Ollama drops part of it server-side without telling the client, which would
# silently discard tool results. A README follow-up (system prompt, history,
# one 20,000-char tool result, answer) is roughly 8-10k tokens. Larger values
# use more memory.
DEFAULT_OLLAMA_NUM_CTX = 16_384


class ProviderError(Exception):
    """Base class for all provider failures."""


class ProviderConnectionError(ProviderError):
    """The inference server could not be reached."""


class ProviderTimeoutError(ProviderError):
    """The inference server did not answer in time."""


class ProviderResponseError(ProviderError):
    """The server returned an error or a malformed response."""


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model (untrusted until validated)."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


class ChatProvider(ABC):
    """Minimal interface the orchestrator needs from an LLM backend."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        """Send a conversation (and optional tool definitions) to the model."""


def parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    """Parse the ``tool_calls`` list of an Ollama assistant message."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ProviderResponseError("'tool_calls' is not a list.")

    calls: list[ToolCall] = []
    for item in raw:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            raise ProviderResponseError("Malformed tool call entry.")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ProviderResponseError("Tool call has no function name.")

        arguments = function.get("arguments", {})
        if arguments is None or arguments == "":
            arguments = {}
        if isinstance(arguments, str):
            # Some servers send arguments as a JSON string.
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ProviderResponseError(
                    f"Tool call arguments for '{name}' are not valid JSON."
                ) from exc
        if not isinstance(arguments, dict):
            raise ProviderResponseError(
                f"Tool call arguments for '{name}' are not an object."
            )

        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)


def parse_chat_response(payload: Any) -> ProviderResponse:
    """Parse a non-streaming Ollama ``/api/chat`` response body."""
    if not isinstance(payload, dict):
        raise ProviderResponseError("Response body is not a JSON object.")

    error = payload.get("error")
    if error:
        raise ProviderResponseError(f"Model error: {str(error)[:300]}")

    message = payload.get("message")
    if not isinstance(message, dict):
        raise ProviderResponseError("Response has no 'message' object.")

    content = message.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ProviderResponseError("Message content is not a string.")

    return ProviderResponse(
        content=content,
        tool_calls=parse_tool_calls(message.get("tool_calls")),
    )


class OllamaProvider(ChatProvider):
    """Local inference through Ollama's ``/api/chat`` endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        timeout: float = 180.0,
        temperature: float = 0.0,
        seed: int | None = 0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        num_ctx: int | None = DEFAULT_OLLAMA_NUM_CTX,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Ollama base URL must be http(s)://host[:port].")
        if num_ctx is not None and num_ctx < 1:
            raise ValueError("num_ctx must be >= 1 or None (server default).")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self._options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            self._options["seed"] = seed
        if num_ctx is not None:
            self._options["num_ctx"] = num_ctx

    def _post_json(self, payload: dict[str, Any]) -> Any:
        """Blocking HTTP POST; run via ``asyncio.to_thread``."""
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                # Bounded read: a runaway or hostile server cannot make us
                # buffer an unbounded response.
                body = r.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise ProviderResponseError(
                        "Ollama response exceeded the "
                        f"{self.max_response_bytes}-byte limit."
                    )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                parsed = json.loads(exc.read())
                detail = str(parsed.get("error", ""))[:300]
            except Exception:  # noqa: BLE001
                pass
            raise ProviderResponseError(
                f"Ollama returned HTTP {exc.code}. {detail}".strip()
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderTimeoutError(
                f"Ollama did not respond within {self.timeout}s."
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ProviderTimeoutError(
                    f"Ollama did not respond within {self.timeout}s."
                ) from exc
            raise ProviderConnectionError(
                f"Cannot reach Ollama at {self.base_url}: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise ProviderConnectionError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderResponseError(
                "Ollama returned a response that is not valid JSON."
            ) from exc

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": self._options,
        }
        if tools:
            payload["tools"] = list(tools)

        body = await asyncio.to_thread(self._post_json, payload)
        return parse_chat_response(body)
