"""OpenAI-compatible protocol provider: OpenAI / DeepSeek / Moonshot / Qwen / Zhipu / custom base_url.

Request and response are already OpenAI format; pass through directly (only the model name is replaced);
streaming forwards upstream SSE bytes line by line.
"""

from __future__ import annotations

from typing import AsyncIterator

import asyncio
import json
import time

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

    async def _media_gen(self, model: str, body: dict, api_key: str, path: str) -> "httpx.Response":
        """Media generation passthrough (OpenAI-compatible /images/generations,
        /videos/generations). Endpoint path is configurable per provider."""
        url = f"{self.base_url}/{path}"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = dict(body)
        payload["model"] = model
        return await get_client().post(url, json=payload, headers=headers, timeout=self.timeout)

    async def image_gen(self, model: str, body: dict, api_key: str) -> "httpx.Response":
        return await self._media_gen(model, body, api_key, self.image_path)

    async def video_gen(self, model: str, body: dict, api_key: str) -> "httpx.Response":
        """Video generation with async-task polling.

        Synchronous upstreams return {data:[{video_url}]} immediately. Async
        upstreams (e.g. agnes) return a task object {task_id, status:queued};
        we poll GET {video_path}/{task_id} until completed and wrap the result.
        """
        url = f"{self.base_url}/{self.video_path}"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = dict(body)
        payload["model"] = model
        client = get_client()
        resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code >= 400:
            return resp
        data = resp.json()
        if _extract_media_url(data, "video"):
            return resp  # synchronous result
        task_id = data.get("task_id") or data.get("id")
        if task_id:
            poll_url = f"{url}/{task_id}"
            gh = {"Authorization": f"Bearer {api_key}"}
            deadline = time.monotonic() + self.video_max_wait
            while time.monotonic() < deadline:
                await asyncio.sleep(self.video_poll_interval)
                try:
                    r = await client.get(poll_url, headers=gh, timeout=self.timeout)
                except Exception:
                    continue
                if r.status_code >= 400:
                    continue
                d = r.json()
                status = str(d.get("status") or "").lower()
                if status in ("completed", "succeeded", "success"):
                    vurl = _extract_media_url(d, "video")
                    return httpx.Response(200, json={"data": [{"video_url": vurl or ""}], "task": d},
                                          headers={"content-type": "application/json"})
                if status in ("failed", "error", "canceled"):
                    return httpx.Response(500, json={"error": {"message": f"video task {status}"}})
            return httpx.Response(504, json={"error": {"message": "video task timed out"}})
        return resp

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


def _extract_media_url(d, kind: str) -> str:
    if not isinstance(d, dict):
        return ""
    item = (d.get("data") or [{}])[0] if isinstance(d.get("data"), list) else {}
    md = d.get("metadata") or {}
    if kind == "image":
        cands = [item.get("url"), item.get("b64_json"), d.get("url")]
    else:
        cands = [item.get("video_url"), item.get("url"), d.get("video_url"), d.get("url"), md.get("url")]
    for c in cands:
        if c:
            return c
    return ""
