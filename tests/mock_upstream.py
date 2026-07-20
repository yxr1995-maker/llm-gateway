"""Local mock upstream (for tests): adds /v1/responses and /v1/embeddings endpoints on top of the original.

Original: local mock upstream simulating openai_like / anthropic / gemini on one port.

- POST /v1/chat/completions          openai_like（Authorization: Bearer <key>）
- POST /v1/messages                  anthropic Messages API（x-api-key）
- POST /v1beta/models/<m>:generateContent        gemini non-stream (?key=)
- POST /v1beta/models/<m>:streamGenerateContent  gemini streaming (?alt=sse&key=)
- GET  /v1/models /v1beta/models     health-check probe

Fault injection: each provider's first key returns 500 (to verify failover):
  sk-fake1 / sk-ant-fake1 / AIzaFake1 -> 500
Streaming: sends chunk by chunk with a 0.2s interval.
"""
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHUNK_INTERVAL = 0.2
BAD_KEYS = {"sk-fake1", "sk-ant-fake1", "AIzaFake1"}
PORT = 9100

OPENAI_WORDS = ["Hello", " from", " mock", " openai", "!"]
ANTHROPIC_WORDS = ["Hi", " from", " mock", " claude", "!"]
GEMINI_WORDS = ["Yo", " from", " mock", " gemini", "!"]


def openai_chunk(cid, created, model, delta=None, finish=None):
    obj = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    return ("data: " + json.dumps(obj) + "\n\n").encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silent
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _log(self, msg):
        with open("/tmp/mock-requests.log", "a") as f:
            f.write(f"{time.time():.3f} {msg}\n")

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, events):
        """events: list of bytes (one SSE event block each), sent one by one with a 0.2s interval."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # no Content-Length: the close ends the response (HTTP/1.0-style delimiter)
        self.send_header("Connection", "close")
        self.end_headers()
        for ev in events:
            self.wfile.write(ev)
            self.wfile.flush()
            time.sleep(CHUNK_INTERVAL)
        self.close_connection = True

    # ------------------------------------------------------------- GET (health check)
    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path.startswith("/v1beta/models"):
            self._send_json(200, {"object": "list", "data": []})
        else:
            self._send_json(404, {"error": "not found"})

    # ------------------------------------------------------------- POST
    def do_POST(self):
        body = self._read_body()
        stream = bool(body.get("stream"))

        if self.path == "/v1/chat/completions":
            return self._openai(body, stream)
        if self.path == "/v1/responses":
            return self._responses(body, stream)
        if self.path == "/v1/embeddings":
            return self._embeddings(body)
        if self.path == "/v1/images/generations":
            return self._media(body, "image")
        if self.path == "/v1/videos/generations":
            return self._media(body, "video")
        if self.path == "/v1/messages":
            return self._anthropic(body, stream)
        m = re.match(r"^/v1beta/models/([^:]+):(generateContent|streamGenerateContent)(?:\?(.*))?$",
                     self.path)
        if m:
            model, action, query = m.group(1), m.group(2), m.group(3) or ""
            return self._gemini(body, model, action == "streamGenerateContent", query)
        self._send_json(404, {"error": f"unknown path {self.path}"})


    # ------------------------------------------------------------- responses
    def _responses(self, body, stream):
        auth = self.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if "boom" in str(body.get("model", "")) or key in BAD_KEYS:
            return self._send_json(500, {"error": {"message": f"responses key {key} failed"}})
        model = body.get("model", "?")
        text = "".join(OPENAI_WORDS)
        if not stream:
            return self._send_json(200, {
                "id": "resp_mock1", "object": "response", "created_at": int(time.time()),
                "status": "completed", "model": model,
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            })

        def ev(t, payload):
            return ("event: " + t + "\ndata: " + json.dumps(payload) + "\n\n").encode()

        events = [ev("response.created", {"type": "response.created",
                 "response": {"id": "resp_mock1", "object": "response",
                              "status": "in_progress", "model": model}})]
        for w in OPENAI_WORDS:
            events.append(ev("response.output_text.delta",
                             {"type": "response.output_text.delta", "delta": w}))
        events.append(ev("response.completed", {
            "type": "response.completed",
            "response": {"id": "resp_mock1", "object": "response", "status": "completed",
                         "model": model,
                         "output": [{"type": "message", "role": "assistant",
                                     "content": [{"type": "output_text", "text": text}]}],
                         "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}))
        self._send_sse(events)

    # ------------------------------------------------------------- media generation
    def _media(self, body, kind):
        auth = self.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if key in BAD_KEYS:
            return self._send_json(500, {"error": {"message": f"key {key} exploded"}})
        if kind == "image":
            return self._send_json(200, {"created": int(time.time()),
                "data": [{"url": "http://mock/image.png", "revised_prompt": str(body.get("prompt", ""))}]})
        return self._send_json(200, {"data": [{"video_url": "http://mock/video.mp4"}]})

    # ------------------------------------------------------------- embeddings
    def _embeddings(self, body):
        auth = self.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if key in BAD_KEYS:
            return self._send_json(500, {"error": {"message": f"key {key} exploded"}})
        model = body.get("model", "?")
        return self._send_json(200, {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": model,
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        })

    # ------------------------------------------------------------- openai_like
    def _openai(self, body, stream):
        auth = self.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if "boom" in str(body.get("model", "")):
            return self._send_json(500, {"error": {"message": "model boom always fails"}})
        if "tool-bad" in str(body.get("model", "")):
            # intentionally returns invalid arguments, to verify the gateway's tool repair
            bad = {"id": "chatcmpl-bad", "object": "chat.completion", "created": int(time.time()),
                   "model": body.get("model", "?"),
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "", "type": "function",
                                       "function": {"name": "do_it", "arguments": "not json {"}}]},
                       "finish_reason": "tool_calls"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
            if not stream:
                return self._send_json(200, bad)
            cid, created = "chatcmpl-bad", int(time.time())
            events = [openai_chunk(cid, created, body.get("model","?"), {"role": "assistant"})]
            events.append(openai_chunk(cid, created, body.get("model","?"),
                {"tool_calls": [{"index": 0, "id": "", "type": "function",
                                 "function": {"name": "do_it", "arguments": "not json {"}}]}))
            events += [openai_chunk(cid, created, body.get("model","?"), {}, "tool_calls"), b"data: [DONE]\n\n"]
            return self._send_sse(events)
        if "stream-break" in str(body.get("model", "")) and stream:
            # streaming: send one chunk then simulate an upstream break
            cid, created = "chatcmpl-brk", int(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(openai_chunk(cid, created, body.get("model","?"), {"content": "par"}))
            self.wfile.flush()
            raise ConnectionError("simulated upstream break")
        if key in BAD_KEYS:
            return self._send_json(500, {"error": {"message": f"key {key} exploded",
                                                   "type": "server_error"}})
        model = body.get("model", "?")
        if not stream:
            return self._send_json(200, {
                "id": "chatcmpl-mock1", "object": "chat.completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": "".join(OPENAI_WORDS)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })
        cid, created = "chatcmpl-mock1", int(time.time())
        events = [openai_chunk(cid, created, model, {"role": "assistant"})]
        events += [openai_chunk(cid, created, model, {"content": w}) for w in OPENAI_WORDS]
        events += [openai_chunk(cid, created, model, {}, "stop"), b"data: [DONE]\n\n"]
        self._send_sse(events)

    # ------------------------------------------------------------- anthropic
    def _anthropic(self, body, stream):
        key = self.headers.get("x-api-key", "")
        if key in BAD_KEYS:
            return self._send_json(500, {"type": "error",
                                         "error": {"type": "api_error",
                                                   "message": f"key {key} exploded"}})
        model = body.get("model", "?")
        if not stream:
            return self._send_json(200, {
                "id": "msg_mock1", "type": "message", "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "".join(ANTHROPIC_WORDS)}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 7},
            })
        def ev(etype, payload):
            return ("event: " + etype + "\ndata: " + json.dumps(payload) + "\n\n").encode()
        events = [
            ev("message_start", {"type": "message_start",
                                 "message": {"id": "msg_mock1", "type": "message",
                                             "role": "assistant", "model": model,
                                             "usage": {"input_tokens": 12, "output_tokens": 1}}}),
            ev("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}}),
        ]
        events += [ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                              "delta": {"type": "text_delta", "text": w}})
                   for w in ANTHROPIC_WORDS]
        events += [
            ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ev("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                 "usage": {"output_tokens": 7}}),
            ev("message_stop", {"type": "message_stop"}),
        ]
        self._send_sse(events)

    # ------------------------------------------------------------- gemini
    def _gemini(self, body, model, is_stream, query):
        key = re.search(r"(?:^|&)key=([^&]*)", query)
        key = key.group(1) if key else ""
        if key in BAD_KEYS:
            return self._send_json(500, {"error": {"code": 500,
                                                   "message": f"key {key} exploded",
                                                   "status": "INTERNAL"}})
        if not is_stream:
            return self._send_json(200, {
                "candidates": [{
                    "content": {"role": "model",
                                "parts": [{"text": "".join(GEMINI_WORDS)}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 6,
                                  "totalTokenCount": 14},
            })
        events = []
        for w in GEMINI_WORDS:
            events.append(("data: " + json.dumps({
                "candidates": [{"content": {"role": "model", "parts": [{"text": w}]},
                                "index": 0}],
            }) + "\n\n").encode())
        events.append(("data: " + json.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]},
                            "index": 0, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 6,
                              "totalTokenCount": 14},
        }) + "\n\n").encode())
        self._send_sse(events)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"mock upstream on :{PORT}", flush=True)
    srv.serve_forever()
