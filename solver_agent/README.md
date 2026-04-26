# Math Solver Agent

基于 Hermes Agent 二次封装的**多模型协同解题服务**。通过 FastAPI HTTP 接口对外提供单轮解题能力：内部 Agent 会按照 `multi-model-math-solving` skill 的策略，并行调用 DeepSeek / Qwen / Gemma 三个模型解题，再交叉验证、迭代修正，最终输出最优解。

与本仓库其它入口（CLI、TUI、Gateway）相互独立 — 走自己的 `HERMES_HOME`，**不读也不写用户的 `~/.hermes/config.yaml`**。

---

## 1. 目录结构

```
solver_agent/
├── hermes_home/
│   └── config.yaml                       # 隔离配置：mcp_servers / custom_providers / default_skills
├── skills/
│   └── multi-model-math-solving/SKILL.md # 解题流程 skill
├── traces/                               # 每次请求的全链路 trace JSON（自动生成）
├── api.py                                # FastAPI 应用：POST /solve、GET /health
├── logging.yaml                          # 服务端日志配置
├── run_server.py                         # 启动入口
├── solver.py                             # 核心：solve(problem) -> dict
├── solver_mcp_server.py                  # MCP 封装：暴露 solve_math_problem / verify_analysis
└── README.md                             # 本文件
```

---

## 2. 前置条件

| 依赖                                       | 说明 |
|------------------------------------------|------|
| Python ≥ 3.12                            | Hermes 主仓库已要求 |
| Hermes 主仓库依赖                             | 按本仓 `pyproject.toml` 安装一次即可（含 `mcp` SDK） |
| `fastapi`、`uvicorn[standard]`、`pydantic` | 通常随 Hermes web 路径已装；缺则补装 |
| **gemma4 端点可达**                          | `http://172.168.80.17:3001/v1/`，OpenAI 兼容 |
| **solver MCP server 可达**                 | `solver_agent/hermes_home/config.yaml` 里配置的 Python 解释器、`solver_agent/solver_mcp_server.py` 与后端 `SOLVER_API_BASE` 都要可用 |

补装（仅当缺失时）：

```bash
pip install "fastapi" "uvicorn[standard]" "pydantic"
```

---

## 3. 配置说明

**默认就跑通**，仅当端点 / 路径 / 密钥变了才需要改。所有配置只在 `solver_agent/hermes_home/config.yaml`，与用户主配置无关。

### 3.1 切换大脑模型 / 端点

```yaml
model:
  default: gemma4              # ← 改这里换默认模型
  provider: custom
  base_url: http://172.168.80.17:3001/v1/

custom_providers:
  - name: "172.168.80.17:3001"
    base_url: http://172.168.80.17:3001/v1/
    api_key: mysecurekey123
    model: gemma4
```

### 3.2 切换 / 增减 MCP 服务

```yaml
mcp_servers:
  solver:
    command: /abs/path/to/python              # ← Python 解释器
    args:
      - /abs/path/to/hermes-agent/solver_agent/solver_mcp_server.py
    env:
      SOLVER_API_BASE: http://172.168.80.46:8000  # ← 后端 API 地址
      SOLVER_SOLVE_TIMEOUT: '600'
      SOLVER_VERIFY_TIMEOUT: '300'
    timeout: 660
    connect_timeout: 30
```

> MCP 服务在 Agent 启动时自动连接，**不需要**显式加进 `platform_toolsets`。

### 3.3 关闭 / 替换 skill

```yaml
default_skills:
  - multi-model-math-solving   # 想加多个就在这里追加
```

如要新增 skill，在 `solver_agent/skills/<skill-name>/SKILL.md` 放一份，frontmatter `name:` 与目录名保持一致即可。

---

## 4. 启动

### 4.1 直接运行

```bash
cd solver_agent
python run_server.py
```

> 当前版本的 `run_server.py` 会从**当前工作目录**读取 `logging.yaml`，所以最稳妥的启动方式是先 `cd solver_agent`。

默认监听 `0.0.0.0:8765`。

### 4.2 自定义端口 / 监听地址

```bash
cd solver_agent
SOLVER_HOST=127.0.0.1 SOLVER_PORT=9000 python run_server.py
```

### 4.3 用 systemd / supervisord 常驻

`run_server.py` 是可以直接给进程管理器接管的入口，无需 wrapper。示例 systemd unit：

```ini
[Unit]
Description=Math Solver Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/hermes-agent/solver_agent
ExecStart=/usr/bin/python run_server.py
Environment=SOLVER_HOST=0.0.0.0
Environment=SOLVER_PORT=8765
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## 5. 接口

### `GET /health`

健康检查。

```bash
curl http://localhost:8765/health
# {"status":"ok"}
```

### `POST /solve`

请求体：

```json
{
  "problem": "解方程 2x^2 - 5x + 2 = 0",
  "quiet": false
}
```

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `problem` | string，**必填** | — | 题目文本 |
| `quiet` | bool | `false` | 是否抑制内部 Agent 的 stdout/stderr 与调用日志；排查并发/skill 问题时建议保持 `false` |

响应体：

```json
{
  "answer": "...",            // 最终自然语言解析
  "elapsed_seconds": 87.4,
  "model": "gemma4",
  "trace_id": "trc_a1b2c3d4e5f6"
}
```

> `trace_id` 对应 `solver_agent/traces/<trace_id>.json`，包含完整的事件时间线和原始对话记录，可用于还原解题/校验/修正全过程。

> ⚠️ **单题耗时 1–10 分钟**（三模型并行 + 交叉验证 + 可能的迭代修正）。HTTP 客户端务必把读取超时拉到 ≥ 600s。

### 调用示例

**curl**

```bash
curl -X POST http://localhost:8765/solve \
  -H "Content-Type: application/json" \
  --max-time 700 \
  -d '{"problem": "已知 sin(α + π/4) = 3/5, α ∈ (0, π/2), 求 cos(2α)"}'
```

**Python**

```python
import requests

r = requests.post(
    "http://localhost:8765/solve",
    json={"problem": "已知 sin(α+π/4)=3/5, α∈(0,π/2), 求 cos(2α)"},
    timeout=700,
)
print(r.json()["answer"])
```

**JS / fetch**

```js
const r = await fetch("http://localhost:8765/solve", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({problem: "..."}),
  signal: AbortSignal.timeout(700_000),
});
console.log((await r.json()).answer);
```

---

## 6. 验证流程是否生效

启动后随便发一题，**观察日志**应能看到：

| 信号 | 含义 |
|------|------|
| `Preloaded skills: ['multi-model-math-solving']` | skill 加载 OK |
| `MCP: registered N tool(s) from 1 server(s)` | MCP server 连上、工具注册成功 |
| 三个 `mcp_solver_solve_math_problem` 在同一回合并发触发 | skill 流程被严格遵循 |
| `verify_analysis` 调用，平台与解题平台不同 | 交叉验证按预期跑 |

如果只看到一个 `mcp_solver_solve_math_problem` 调用而不是三个，先确认请求里 `quiet=false`，这样能看到主 Agent 的工具调用日志；若日志里仍只有一个 tool call，再考虑把 `model.default` 换强一点，或在题目前明确加一句"按 multi-model-math-solving 流程执行"。

---

## 7. 故障排查

| 症状 | 原因 / 处理 |
|------|------|
| 启动时报 `Failed to connect to MCP server 'solver'` | `command` 路径错 / 解释器没装 mcp 包 / 后端 `SOLVER_API_BASE` 不可达 |
| `Skill(s) not found: ['multi-model-math-solving']` | `solver_agent/skills/multi-model-math-solving/SKILL.md` 缺失，或 frontmatter `name:` 与目录名不一致 |
| 500 + `... unauthorized ...` | `custom_providers[0].api_key` 与 gemma4 端点不匹配 |
| Agent 串行而非并行调三模型 | 先确认日志里是否出现 3 条 `agent tool start: mcp_solver_solve_math_problem ...`；若只有 1 条，再把 `config.yaml` 的 `agent.reasoning_effort: medium` 提到 `high` |
| Compression model (gemma4) context is 65,434... | 压缩模型窗口略小于主模型压缩阈值；把 `hermes_home/config.yaml` 的 `compression.threshold` 从 `0.50` 调到 `0.49` |
| HTTP 客户端 504 / read timeout | 客户端超时 < 解题耗时；把客户端 timeout 拉到 600s+ |
| `mcp_servers` 未生效（工具没注册） | 检查 `platform_toolsets.cli` 没有写 `no_mcp`；MCP 在非空时是自动并入的 |
| 多请求时第二个卡住 | MCP stdio 子进程通常单连接，多并发会排队 — 可调小 `ThreadPoolExecutor(max_workers=...)` 或换 HTTP transport |

---

## 8. Trace 全链路追踪

每次 `/solve` 请求会自动在 `solver_agent/traces/` 下生成一份 JSON 文件（无论 `quiet` 是否为 `true`）。

文件名格式：`trc_<12位hex>.json`，与响应里的 `trace_id` 对应。

trace 文件包含：

| 字段 | 说明 |
|------|------|
| `request` | 原始请求参数（题目、模型、provider） |
| `result` | 最终答案、耗时、token 用量 |
| `events` | 按时间排序的回调事件流：`tool_start` / `tool_complete` / `step` / `status` / `text` / `clarify` |
| `messages` | AIAgent 完整对话记录（user / assistant / tool 所有角色的原始消息） |

排查示例：
- 看 `events` 里的 `tool_start` / `tool_complete` 配对，确认三模型是否并行、各自耗时多少
- 看 `messages` 里 tool role 的 content，拿到每个模型的完整解析和验证结果
- 看 `events` 里的 `status` 事件，定位 ReadTimeout / 重连等异常

trace 目录可通过 `SOLVER_TRACE_DIR` 环境变量覆盖。

---

## 9. 已知约束 / 后续扩展

- **单轮 + 同步**：当前接口不流式、不维护多轮上下文。如需，建议加 `POST /sessions` + `POST /sessions/{id}/turn` 走 SSE。
- **无鉴权**：内网部署直接用；若要外网暴露，加一个 `Depends(check_api_key)` 校验 `Authorization` 头即可。
- **配置热更新**：当前修改 `hermes_home/config.yaml` 后必须重启进程才能生效。
- **多 skill 串联**：`default_skills` 列表里追加更多 skill 即可被同时预加载。
