"""Shared HTTP plumbing for provider integrations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from aire.core.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def map_http_error(provider: str, exc: httpx.HTTPStatusError) -> ProviderError:
    """Translate an HTTP status failure into the structured error hierarchy."""
    response = exc.response
    status = response.status_code
    detail = ""
    try:
        payload = response.json()
        error = payload.get("error", payload)
        detail = error.get("message", "") if isinstance(error, dict) else str(error)
    except (json.JSONDecodeError, ValueError):
        detail = response.text[:500]
    message = detail or f"HTTP {status}"
    if status in {401, 403}:
        return AuthenticationError(provider, message, status=status, cause=exc)
    if status == 429:
        return RateLimitError(provider, message, status=status, cause=exc)
    return ProviderError(provider, message, status=status, cause=exc)


class ProviderHttpClient:
    """An httpx.AsyncClient registered on the runtime for lifecycle cleanup."""

    def __init__(
        self,
        runtime: Runtime,
        provider: str,
        *,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    ) -> None:
        self.provider = provider
        self._client = runtime.resources.track_resource(
            f"http:{provider}",
            httpx.AsyncClient(base_url=base_url, headers=headers or {}, timeout=timeout),
        )

    @property
    def raw(self) -> httpx.AsyncClient:
        return self._client

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise map_http_error(self.provider, exc) from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"{self.provider} request timed out", context={"path": path}, cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.provider, str(exc), cause=exc) from exc
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                self.provider, f"non-JSON response from {path}", retryable=False, cause=exc
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(self.provider, "unexpected response shape", retryable=False)
        return data

    async def get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise map_http_error(self.provider, exc) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.provider, str(exc), cause=exc) from exc
        if not isinstance(data, dict):
            raise ProviderError(self.provider, "unexpected response shape", retryable=False)
        return data

    async def stream_sse(self, path: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed JSON objects from a server-sent-events stream."""
        try:
            async with self._client.stream("POST", path, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise map_http_error(
                        self.provider,
                        httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        ),
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        return
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        yield parsed
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"{self.provider} stream timed out", context={"path": path}, cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.provider, str(exc), cause=exc) from exc
