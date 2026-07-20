# llm-gateway

> A self-hosted, OpenAI-compatible LLM gateway. Aggregate multiple providers (OpenAI / Anthropic / Gemini / DeepSeek / any OpenAI-compatible service) behind one API with model aliases, a key pool with automatic failover, usage stats, rate limiting, a built-in admin console, plus Mixture-of-Agents and planner-worker pipelines (incl. multimodal: vision / image-gen / video-gen).

[简体中文](README.zh-CN.md)

A single-process FastAPI service that exposes an **OpenAI-compatible API** and forwards requests to multiple LLM providers. Typical use case: point a client like Codex at the gateway via `OPENAI_BASE_URL`, then switch the underlying model freely with **model aliases** (e.g. `claude`, `ds`), backed by a rotating key pool with failover, usage statistics, and rate limiting.

## Features

- **OpenAI-compatible**: `POST /v1/chat/completions` (with SSE streaming), `GET /v1/models`
- **Responses passthrough**: `POST /v1/responses` (with streaming) for Responses-only clients such as the Codex CLI
- **Embeddings passthrough**: `POST /v1/embeddings`, sharing the same key pool and stats
- **Anthropic Messages input**: `POST /v1/messages`, so Anthropic-protocol clients can also reach any upstream
- **Unified routing**: Responses is the canonical hub — any input face (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) can reach any upstream (openai_like / anthropic / gemini), with cross-protocol streaming conversion
- **MOA (Mixture-of-Agents)**: parallel proposers + aggregator synthesis, **or a staged multimodal pipeline** (text / vision / image-gen / video-gen) that chains models into a hybrid LLM; trigger with `model: moa:<name>`
- **Planner-Worker**: a strong planner model decomposes the task and cheap worker models execute subtasks in parallel, then the planner synthesizes (optionally re-plans); trigger with `model: pw:<name>` — independent of MOA
- **Media generation**: OpenAI-compatible image/video generation with async-task polling; mix generative models into pipelines
- **Tool resilience**: auto-repairs malformed `tool_call` JSON; if the upstream errors or drops mid-stream, a valid terminator (`[DONE]` / `response.completed` / `message_stop`) is synthesized so the client session never breaks on a broken stream
- **Multi-provider**: OpenAI-compatible, Anthropic Messages, Google Gemini — automatic format conversion
- **Model aliases**: short alias -> `provider/model`, transparent switching
- **Key pool**: round-robin across multiple keys, automatic cooldown and failover on failure
- **Usage stats**: SQLite-persisted (`data/usage.db`), aggregated by model / provider
- **Rate limiting**: in-memory token bucket per caller key
- **Admin console**: zero-build single-file UI for config editing, usage dashboard, and model testing

## Quick start

```bash
# 1. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2. Prepare config
cp config.example.yaml config.yaml
#    edit config.yaml and fill in real API keys for each provider

# 3. Run
python -m app.main
# equivalent to: uvicorn app.main:app --host 0.0.0.0 --port 8080
```

After startup:

- API base: `http://127.0.0.1:8080/v1`
- Admin console: `http://127.0.0.1:8080/` (root redirects to the console)

Verify:

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer sk-local-xxxx"

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-local-xxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "ds", "messages": [{"role": "user", "content": "hello"}]}'
```

Add `"stream": true` to the body for SSE streaming (`data: {...}`, terminated by `data: [DONE]`).

## Configuration (config.yaml)

```yaml
server:
  host: 0.0.0.0                   # listen address
  port: 8080                      # listen port
  master_key: "sk-local-xxxx"     # gateway/admin key; empty = no auth (local-only)

providers:                        # one entry per provider
  <name>:
    type: openai_like             # openai_like / anthropic / gemini
    base_url: https://...         # upstream URL (see cheat sheet below)
    keys: ["sk-...", "sk-..."]    # key pool, round-robin + failover
    models: ["model-a", ...]      # available models
    supports_responses: false     # true if upstream natively supports /v1/responses (passthrough fast path)

aliases:                          # alias -> provider/model
  claude: anthropic/claude-sonnet-4-5

rate_limit:
  requests_per_minute: 60         # per-caller-key limit; 0 = unlimited
```

Model resolution (the `model` field in requests):

1. **Alias**: matches `aliases`, routes to `provider/model`
2. **provider/model**: explicit
3. **Bare model name**: matches a configured provider's `models` list

The config file hot-reloads on save — no restart needed.

## Codex integration

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-local-xxxx     # the master_key from config.yaml

codex --model claude                    # use an alias directly
```

`model` can be any alias, or a `provider/model` full name (e.g. `deepseek/deepseek-reasoner`). Switching models is just changing the alias — no need to juggle per-provider base URLs and keys.

## Unified multi-protocol routing

Responses is the canonical hub. Pick any input face; all reach any upstream:

| Endpoint | Input format | Internal path |
| --- | --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat | direct (chat<->native conversion built in) |
| `POST /v1/responses` | OpenAI Responses | passthrough if `supports_responses`; else responses->chat->native->chat->responses |
| `POST /v1/messages` | Anthropic Messages | anthropic->chat->native->chat->anthropic |

`supports_responses: true` (per provider) marks an upstream that natively speaks Responses and takes the passthrough fast path; others are bridged through chat. Streaming is converted across protocols too.

## Mixture-of-Agents (MOA)

Define a pipeline in `config.yaml`:

```yaml
moa:
  default:
    proposers:
      - provider: openai
        model: gpt-4o
      - provider: anthropic
        model: claude-sonnet-4-5
    aggregator:
      provider: openai
      model: gpt-4o
    # aggregator_prompt: "custom synthesis prompt (optional)"
```

Set `model` to `moa:default` (or `moa/default`): proposers answer in parallel -> aggregator synthesizes -> returned. All three input faces support it; streaming outputs the aggregator's synthesis. A single proposer failure is recorded as a note and skipped — it never breaks the whole run.

## Planner-Worker (pw)

Independent of MOA. A strong **planner** model decomposes the task into subtasks; cheap **worker** models execute them in parallel; the planner synthesizes a final answer (and may re-plan for up to `max_rounds`). Expensive tokens go to planning, grunt work to cheap models.

```yaml
planner_worker:
  solve:
    planner: { provider: openai, model: gpt-4o, reasoning_effort: high }
    workers:
      - { provider: deepseek, model: deepseek-chat, reasoning_effort: low }
      - { provider: openai, model: gpt-4o-mini }
    max_rounds: 1
    max_concurrency: 4
```

Trigger with `model: pw:solve`. Worker failures are skipped (noted), never aborting the run. Reuses the TPS scheduler (concurrency cap, parallel dispatch, streamed final synthesis).

## Admin console

Open `http://127.0.0.1:8080/` in a browser:

> Entering the console requires no auth by default. A toggle in the page lets you turn the `master_key` on/off freely (when off, both the console and `/v1/*` are unauthenticated — local only). The console can add/edit/delete providers, models, aliases, MOA pipelines, and rate limits; changes hot-reload on save.

1. **Config editor**: edit providers / models / aliases / MOA / rate limit / auth toggle; keys are stored only in the local `config.yaml`
2. **Overview**: masked provider/alias tables, one-click health check
3. **Usage dashboard**: calls, tokens, latency, success rate over 1/7/30 days, plus the last 50 calls
4. **Model test**: pick a model -> send "Say hi in one word" -> see latency and reply

Admin API (require `Authorization: Bearer <master_key>` when auth is on):

| Endpoint | Description |
| --- | --- |
| `GET /admin/api/config` | current effective config (masked) |
| `GET /admin/api/config/raw` | full config (unmasked, for the editor) |
| `PUT /admin/api/config` | save full config (hot-reloads) |
| `GET /admin/api/usage/summary?days=7` | usage aggregates |
| `GET /admin/api/usage/recent` | recent calls |
| `GET /admin/api/health` | provider connectivity probe |
| `POST /admin/api/test` | model test, body `{"model": "<alias>"}` |

## Docker

```bash
docker build -t llm-gateway .

docker run -d --name llm-gateway \
  -p 8080:8080 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/data:/app/data \
  llm-gateway
```

- `/app/config.yaml`: config file (**required**; mounted changes hot-reload too)
- `/app/data`: usage DB directory (optional; without it, stats are lost when the container is removed)

## Testing

No real API key needed — the built-in mock upstream runs an end-to-end smoke test (auth, model list, three resolution forms, three provider protocol conversions, streaming, failover, 502, Responses passthrough, embeddings, MOA, config editing, admin API):

```bash
.venv/bin/python tests/smoke_test.py
```

Expect `=== 47/47 passed ===`. The mock simulates openai_like / anthropic / gemini on one port (plus image/video endpoints) and injects a 500 for each provider's first key to verify failover. It also covers MOA (parallel + multimodal stages), planner-worker, tool repair, mid-stream recovery, and admin config editing.

## Run as a macOS service

Register the gateway as a launchd user agent for auto-start + crash recovery. See [`contrib/macos/README.md`](contrib/macos/README.md).

> ⚠️ macOS TCC blocks launchd from executing programs under `~/Documents`, `~/Desktop`, `~/Downloads` (reporting `Operation not permitted`). Place the repo elsewhere (e.g. `~/llm-gateway`).

## Upstream base_url cheat sheet

| Provider | type | base_url |
| --- | --- | --- |
| OpenAI | `openai_like` | `https://api.openai.com/v1` |
| DeepSeek | `openai_like` | `https://api.deepseek.com/v1` |
| Moonshot Kimi | `openai_like` | `https://api.moonshot.cn/v1` |
| Zhipu GLM | `openai_like` | `https://open.bigmodel.cn/api/paas/v4` |
| Alibaba Bailian (Qwen) | `openai_like` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Google Gemini | `gemini` | `https://generativelanguage.googleapis.com/v1beta` |
| Anthropic | `anthropic` | `https://api.anthropic.com` |

> Gemini can also use its OpenAI-compatible endpoint (set `type: openai_like`): `https://generativelanguage.googleapis.com/v1beta/openai/`

## Project structure

```
app/
  main.py            # FastAPI assembly and entry point (root -> console redirect)
  config.py          # YAML config loading + hot reload + save
  router.py          # /v1/chat/completions, /v1/responses, /v1/messages, /v1/models, /v1/embeddings
  protocol.py        # responses<->chat, anthropic<->chat converters (incl. streaming)
  moa.py             # Mixture-of-Agents (parallel proposers + multimodal staged pipeline)
  planner_worker.py  # Planner-Worker pattern (strong planner + cheap workers)
  providers/         # openai_like / anthropic / gemini protocol adapters
  pool.py            # key pool: round-robin + failover
  stats.py           # SQLite usage stats (aiosqlite)
  ratelimit.py       # token-bucket rate limiter
  admin.py           # admin API: /admin/api/*
static/
  admin.html         # admin console (single file, no external deps)
config.example.yaml  # example config
requirements.txt
Dockerfile
tests/               # mock upstream + end-to-end smoke test (no real key needed)
contrib/macos/       # launchd service example (plist + launch script)
```

## Notes

- All upstream calls default to a 60s timeout; health-check probes time out at 5s
- For streaming responses where the upstream returns no usage, tokens are estimated as chars / 4
- `master_key` is the gateway's only credential — use a strong random value; never leave it empty when exposed to a public network
- Usage data is stored in local SQLite for personal observability only; nothing is uploaded

## License

MIT © Eran. See [LICENSE](LICENSE).
