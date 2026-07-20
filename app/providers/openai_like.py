"""OpenAI 兼容协议 provider：OpenAI / DeepSeek / Moonshot / Qwen / 智谱 / 自定义 base_url。

请求与响应均已是 OpenAI 格式，直接透传（仅替换 model 名）；
流式逐行转发上游 SSE 字节。
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from . import ProviderBase, UpstreamError, get_client, register


@register("openai_like")
class OpenAILikeProvider(ProviderBase):
    """OpenAI 兼容透传。"""

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
            # 原样返回上游响应（非 2xx 由路由层判定并故障转移）
            return await get_client().post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        return self._stream(url, payload, headers)

    async def embeddings(self, model: str, body: dict, api_key: str) -> "httpx.Response":
        """向量 embedding 透传（OpenAI 兼容 /v1/embeddings）。"""
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
        """流式透传：边收边发，逐行产出字节（aiter_lines 会去掉换行，逐行补回）。"""
        client = get_client()
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=self.timeout
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread())[:500].decode("utf-8", "replace")
                raise UpstreamError(resp.status_code, detail)
            async for line in resp.aiter_lines():
                # 空行是 SSE 事件分隔符，同样原样补回换行
                yield line.encode("utf-8") + b"\n"

    async def responses(
        self, model: str, body: dict, api_key: str, stream: bool
    ) -> "httpx.Response | AsyncIterator[bytes]":
        """OpenAI Responses API 透传（/responses）。

        请求与响应均已是 Responses 格式，直接透传（仅替换 model 名）。
        仅当上游原生支持 Responses 协议时可用（如火山 Ark / agnes）；
        不支持的协议（Anthropic / Gemini）不在此 provider 提供。
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
