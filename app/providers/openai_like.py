"""OpenAI-compatible protocol provider: OpenAI / DeepSeek / Moonshot / Qwen / Zhipu / custom base_url.

Request and response are already OpenAI format; pass through directly (only the model name is replaced);
streaming forwards upstream SSE bytes line by line.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from . import ProviderBase, UpstreamError, get_client, register


@register("openai_like")
class OpenAILikeProvider(ProviderBase):
    """OpenAI-compatible passthrough."""

    async def chat_completions(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = dict(body)
        payload["model"] = model
        payload["stream"] = bool(stream)

        if not stream:
            # Return the upstream response as-is (non-2xx judged and failed over by the router layer)
            return await get_client().post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        return self._stream(url, payload, headers)

    async def embeddings(self, model: str, body: dict, api_key: str) -> "httpx.Response":
        """Embeddings passthrough (OpenAI-compatible /v1/embeddings)."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = dict(body)
        payload["model"] = model
        return await get_client().post(
            url, json=payload, headers=headers, timeout=self.timeout
        )

    async def _stream(self, url: str, payload: dict, headers: dict) -> AsyncIterator[bytes]:
        """Streaming passthrough: stream as received, yielding bytes line by line (aiter_lines strips newlines, so re-add them)."""
        client = get_client()
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=self.timeout
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread())[:500].decode("utf-8", "replace")
                raise UpstreamError(resp.status_code, detail)
            async for line in resp.aiter_lines():
                # blank lines are SSE event separators; re-add the newline as-is
                yield line.encode("utf-8") + b"\n"

    async def responses(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        """OpenAI Responses API passthrough（/responses）。

        Request and response are already Responses format; pass through directly (only the model name is replaced).
        Available only when the upstream natively supports the Responses protocol (e.g. Volcano Ark / agnes);
        unsupported protocols (Anthropic / Gemini) are not provided by this provider.
        """
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = dict(body)
        payload["model"] = model
        payload["stream"] = bool(stream)

        if not stream:
            return await get_client().post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        return self._stream(url, payload, headers)
