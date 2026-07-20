# llm-gateway

> A personal, self-hosted LLM API aggregation gateway. Exposes an OpenAI-compatible API and forwards requests to multiple LLM providers (OpenAI / Anthropic / Gemini / DeepSeek / and any OpenAI-compatible service) with model aliases, a key pool with automatic failover, usage stats and rate limiting.

个人用 LLM API 聚合网关：单进程 FastAPI 服务，对外暴露 **OpenAI 兼容 API**，把请求转发到多家 LLM 服务商（OpenAI / Anthropic / DeepSeek / Gemini 及任意 OpenAI 兼容服务）。

典型场景：Codex 等客户端只需设置 `OPENAI_BASE_URL` 指向本网关，即可通过**模型别名**（如 `claude`、`ds`）自由切换底层模型，配合密钥池轮询与故障转移、用量统计和限流。

## 功能特性

- **OpenAI 兼容**：`POST /v1/chat/completions`（含 SSE 流式）、`GET /v1/models`
- **Responses 透传**：`POST /v1/responses`（含流式），让只说 Responses API 的客户端（如新版 Codex CLI）也能走网关；仅 openai_like 上游支持
- **Embeddings 透传**：`POST /v1/embeddings`，向量接口同样享受密钥池与统计
- **Anthropic Messages 输入**：`POST /v1/messages`，只说 Anthropic 协议的客户端也能聚合到任意上游
- **MOA 混合智能体**：配置多个 proposer 并行作答 + aggregator 综合，`model` 填 `moa:<name>` 触发
- **工具韧性**：自动修复非法 tool_call JSON、流式中途上游异常或提前断开时合成合法收尾（`[DONE]`/`completed`），避免客户端会话因断流中断
- **Responses API 透传**：`POST /v1/responses`（含 SSE 流式），供 Codex 等
  Responses-only 客户端使用；仅 `openai_like` provider 且上游原生支持
  Responses 协议时可用（如火山 Ark / agnes），不支持的 provider 返回 501
- **多 Provider**：OpenAI 兼容协议、Anthropic Messages、Google Gemini，自动做格式转换
- **模型别名**：`aliases` 配置短别名 → `provider/model`，客户端无感切换
- **密钥池**：多 key 轮询、失败自动冷却与故障转移
- **用量统计**：SQLite 持久化（`data/usage.db`），按模型 / Provider 聚合
- **限流**：按调用方 key 的内存令牌桶
- **管理页**：零构建单文件页面，配置总览 / 用量仪表盘 / 模型测试

## 快速开始

```bash
# 1. 安装依赖（Python 3.11+）
pip install -r requirements.txt

# 2. 准备配置
cp config.example.yaml config.yaml
#    编辑 config.yaml，填入各 provider 的真实 API key

# 3. 启动
python -m app.main
# 等价于：uvicorn app.main:app --host 0.0.0.0 --port 8080
```

启动后：

- API 入口：`http://127.0.0.1:8080/v1`
- 管理页：`http://127.0.0.1:8080/static/admin.html`（或直接访问根路径 `http://127.0.0.1:8080/`，会跳转到管理页）

验证：

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer sk-local-xxxx"

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-local-xxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "ds", "messages": [{"role": "user", "content": "你好"}]}'
```

流式只需在请求体中加 `"stream": true`，响应为标准 SSE（`data: {...}`，结束 `data: [DONE]`）。

## 配置详解（config.yaml）

```yaml
server:
  host: 0.0.0.0                   # 监听地址
  port: 8080                      # 监听端口
  master_key: "sk-local-xxxx"     # 调用网关与管理页的 Key；留空则无需鉴权（仅限本机调试）

providers:                        # 每家服务商一个条目
  <名称>:
    type: openai_like             # openai_like / anthropic / gemini
    base_url: https://...         # 上游地址（见文末速查表）
    keys: ["sk-...", "sk-..."]    # 密钥池，轮询 + 故障转移
    models: ["model-a", ...]      # 对外可用模型清单

aliases:                          # 别名 → provider/model
  claude: anthropic/claude-sonnet-4-5

rate_limit:
  requests_per_minute: 60         # 每个调用 key 的每分钟限额；0 = 不限
```

模型解析规则（请求体 `model` 字段）：

1. **别名**：命中 `aliases`，路由到对应 `provider/model`
2. **provider/model**：显式指定
3. **裸模型名**：匹配已配置 provider 的 `models` 清单

配置文件保存后自动热重载，无需重启服务。

## Codex 接入示例

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-local-xxxx     # 即 config.yaml 里的 master_key

codex --model claude                    # model 直接填别名
```

model 可填 `aliases` 中的任意别名（如 `claude`、`ds`、`gem`），也可用 `provider/model` 全称（如 `deepseek/deepseek-reasoner`）。切换模型只改别名即可，无需再管各家不同的 base_url 与 key。

## 多协议统一路由

网关以 **Responses 为统一枢纽**，三种输入面任选其一，均可到达任意上游：

| 输入端点 | 输入格式 | 内部路径 |
| --- | --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat | 直达（各家 chat↔原生转换已内置） |
| `POST /v1/responses` | OpenAI Responses | 上游原生支持则透传；否则 responses→chat→原生→chat→responses |
| `POST /v1/messages` | Anthropic Messages | anthropic→chat→原生→chat→anthropic |

`supports_responses: true`（per-provider）标记上游原生支持 Responses，走透传快路径；其余上游自动经 chat 桥接转换。流式同样跨协议转换。

## Mixture-of-Agents（MOA）

在 `config.yaml` 定义流水线：

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
    # aggregator_prompt: "自定义综合提示（可选）"
```

请求时 `model` 填 `moa:default`（或 `moa/default`）即可：proposer 并行作答 → aggregator 综合 → 返回。三种输入面都支持（`/v1/chat/completions`、`/v1/responses`、`/v1/messages`），流式会输出 aggregator 的综合过程。单个 proposer 失败会记为注记并继续，不影响整体。

## ## 管理页

浏览器打开 `http://127.0.0.1:8080/static/admin.html`（或根路径）：

1. 首屏输入 `master_key`（仅保存在浏览器 localStorage）
2. **配置总览**：Provider / 模型 / 别名映射表格，密钥已脱敏；可一键健康检查各 Provider 连通性
3. **用量仪表盘**：近 1/7/30 天按模型聚合的调用次数、Token、平均延迟、成功率，以及最近 50 条调用明细
4. **模型测试**：下拉选模型 → 发送 “Say hi in one word” → 显示延迟与回复 / 错误

管理 API（均需 `Authorization: Bearer <master_key>`）：

| 接口 | 说明 |
| --- | --- |
| `GET /admin/api/config` | 当前生效配置（脱敏） |
| `GET /admin/api/usage/summary?days=7` | 用量聚合 |
| `GET /admin/api/usage/recent` | 最近调用明细 |
| `GET /admin/api/health` | 各 Provider 连通性探测 |
| `POST /admin/api/test` | 模型测试，body `{"model": "别名"}` |

## Docker 运行

```bash
docker build -t llm-gateway .

docker run -d --name llm-gateway \
  -p 8080:8080 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/data:/app/data \
  llm-gateway
```

- `/app/config.yaml`：配置文件（**必需**，挂载后改配置同样热生效）
- `/app/data`：用量统计数据库目录（可选，不挂载则容器删除后统计数据丢失）

## 测试

无需真实 API key，用内置 mock 上游跑端到端冒烟测试（覆盖鉴权、模型列表、三种寻址、三种 provider 协议转换、流式、故障转移、502、Responses 透传、embeddings、管理 API）：

```bash
.venv/bin/python tests/smoke_test.py
```

预期输出 `=== 20/20 passed ===`。mock 上游在同一端口模拟 openai_like / anthropic / gemini 三种协议，并对每个 provider 的第一个 key 注入 500 以验证故障转移。

## macOS 常驻运行

用 launchd 把网关注册为用户服务，开机自启 + 崩溃自动拉起。见 [`contrib/macos/README.md`](contrib/macos/README.md)。

> ⚠️ macOS TCC 会阻止 launchd 执行 `~/Documents`、`~/Desktop`、`~/Downloads` 下的程序（报 `Operation not permitted`），请把仓库放在其他位置（如 `~/llm-gateway`）。

## ## 常见上游 base_url 速查表

| 服务商 | type | base_url |
| --- | --- | --- |
| OpenAI | `openai_like` | `https://api.openai.com/v1` |
| DeepSeek | `openai_like` | `https://api.deepseek.com/v1` |
| Moonshot Kimi | `openai_like` | `https://api.moonshot.cn/v1` |
| 智谱 GLM | `openai_like` | `https://open.bigmodel.cn/api/paas/v4` |
| 阿里百炼（通义） | `openai_like` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Google Gemini | `gemini` | `https://generativelanguage.googleapis.com/v1beta` |
| Anthropic | `anthropic` | `https://api.anthropic.com` |

> Gemini 也可使用其 OpenAI 兼容端点（type 填 `openai_like`）：`https://generativelanguage.googleapis.com/v1beta/openai/`

## 项目结构

```
app/
  main.py            # FastAPI 装配与启动入口（含根路径 → 管理页跳转）
  config.py          # YAML 配置加载 + 热重载
  router.py          # /v1/chat/completions、/v1/models
  providers/         # openai_like / anthropic / gemini 协议适配
  pool.py            # 密钥池：轮询 + 故障转移
  stats.py           # SQLite 用量统计（aiosqlite）
  ratelimit.py       # 令牌桶限流
  admin.py           # 管理 API：/admin/api/*
static/
  admin.html         # 管理页（单文件，无外部依赖）
config.example.yaml  # 示例配置
requirements.txt
Dockerfile
tests/               # mock 上游 + 端到端冒烟测试（无需真实 key）
contrib/macos/       # launchd 常驻示例（plist + 启动脚本）
```

## 说明

- 所有上游调用默认 60s 超时；健康检查探测超时 5s
- 用量统计中，流式响应若上游未返回 usage，按字符数 / 4 估算 token
- `master_key` 是网关唯一凭证，请使用强随机值；暴露到公网时切勿留空
- 统计数据保存在本地 SQLite，仅用于个人观测，不会上传任何数据

## License

MIT © Eran。详见 [LICENSE](LICENSE)。
