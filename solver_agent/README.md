# Math Solver Agent

基于 Hermes Agent 二次封装的**多模型协同解题服务**。通过 FastAPI HTTP 接口对外提供单轮解题能力：内部 Agent 按照 `multi-model-math-solving` skill 的策略，并行调用 DeepSeek / Gemma 两个模型解题，再由 Qwen 交叉验证、迭代修正，最终输出最优解。

与本仓库其它入口（CLI、TUI、Gateway）相互独立 — 走自己的 `HERMES_HOME`，**不读也不写用户的 `~/.hermes/config.yaml`**。

---

## 1. 目录结构

```
solver_agent/
├── hermes_home/
│   ├── config.yaml                          # 隔离配置：mcp_servers / custom_providers / default_skills
│   ├── SOUL.md                              # Agent 系统提示词
│   └── logs/                                # 运行时日志（agent.log / errors.log / mcp-stderr.log）
├── skills/
│   └── multi-model-math-solving/SKILL.md    # 解题流程 skill
├── test/
│   └── test_batch_solve_xls.py              # 批量 Excel 解题脚本
├── traces/                                  # 每次请求的全链路 trace JSON（自动生成）
├── api.py                                   # FastAPI 应用：POST /solve、GET /health
├── solver.py                                # 核心：solve(problem) -> dict
├── solver_mcp_server.py                     # MCP 封装：暴露 solve_math_problem / verify_analysis / ping
├── run_server.py                            # 启动入口
├── logging.yaml                             # 服务端日志配置
├── requirements.txt                         # 运行时依赖
└── README.md                                # 本文件
```

---

## 2. 前置条件

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.12 | Hermes 主仓库已要求 |
| Hermes 主仓库依赖 | 按本仓 `pyproject.toml` 安装一次即可（含 `mcp` SDK） |
| `fastapi`、`uvicorn[standard]`、`pydantic` | 通常随 Hermes web 路径已装；缺则补装 |
| **后端解题服务可达** | `SOLVER_API_BASE`（默认 `http://172.168.80.46:8000`） |
| **大脑模型端点可达** | `hermes_home/config.yaml` 中 `custom_providers` 配置的地址 |

一键安装所有依赖（在仓库根目录执行）：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r solver_agent/requirements.txt
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

## 4. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HERMES_HOME` | `solver_agent/hermes_home` | 配置目录，自动设置 |
| `SOLVER_AGENT_SKILLS_DIR` | `solver_agent/skills` | Skill 目录，自动设置 |
| `SOLVER_HOST` | `0.0.0.0` | HTTP 服务监听地址 |
| `SOLVER_PORT` | `8765` | HTTP 服务监听端口 |
| `SOLVER_API_BASE` | `http://172.168.80.46:8000` | 后端解题/校验服务地址 |
| `SOLVER_API_KEY` | _(空)_ | 后端 API 鉴权 key，可选 |
| `SOLVER_SOLVE_TIMEOUT` | `600` | 解题请求超时（秒） |
| `SOLVER_VERIFY_TIMEOUT` | `300` | 校验请求超时（秒） |
| `SOLVER_TRACE_DIR` | `solver_agent/traces` | Trace 文件输出目录 |

---

## 5. 启动

### 5.1 直接运行

```bash
cd solver_agent
python run_server.py
```

> `run_server.py` 会从**当前工作目录**读取 `logging.yaml`，所以最稳妥的启动方式是先 `cd solver_agent`。

默认监听 `0.0.0.0:8765`。

### 5.2 自定义端口 / 监听地址

```bash
cd solver_agent
SOLVER_HOST=127.0.0.1 SOLVER_PORT=9000 python run_server.py
```

### 5.3 用 systemd / supervisord 常驻

`run_server.py` 可以直接给进程管理器接管，无需 wrapper。示例 systemd unit：

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

## 6. 接口

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
| `quiet` | bool | `false` | 是否抑制内部 Agent 的 stdout/stderr 与调用日志 |

响应体：

```json
{
  "answer": "...",
  "elapsed_seconds": 87.4,
  "model": "gemma4",
  "trace_id": "trc_a1b2c3d4e5f6"
}
```

> `trace_id` 对应 `solver_agent/traces/<trace_id>.json`，包含完整的事件时间线和原始对话记录。

> **单题耗时 1–10 分钟**（双模型并行 + 交叉验证 + 可能的迭代修正）。HTTP 客户端务必把读取超时拉到 ≥ 600s。

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

## 7. MCP 工具详情

`solver_mcp_server.py` 通过 stdio 传输暴露三个 MCP 工具，供 Agent 内部调用：

### `solve_math_problem`

使用指定大模型对数学题进行解答。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `text_input` | str | `""` | 题干文本 |
| `image_base64` | str? | — | 题目图片 base64（不带 `data:` 前缀） |
| `url_image` | str? | — | 题目图片 URL |
| `solve_platform` | str | `"DeepSeek"` | 解题平台 |
| `solve_model` | str | `"deepseek-v3.2"` | 解题模型，需与平台匹配 |
| `thinking` | bool | `true` | 是否开启思维链 |
| `verify` | bool | `false` | 是否自动校验并迭代修正 |
| `verify_platform` | str | `"Qwen"` | 校验平台 |
| `verify_model` | str | `"qwen3-235b-a22b"` | 校验模型 |
| `verify_round` | int | `5` | 最大校验轮数 |
| `question_type` | str | `""` | 题型提示（如"选择题"） |
| `prompt` | str | `""` | 自定义解题 prompt |
| `example` | str | `""` | few-shot 示例 |

平台与模型对应关系：

| 平台 | 模型 |
|------|------|
| `DeepSeek` | `deepseek-v3.2` |
| `Qwen` | `qwen3-235b-a22b` |
| `Gemma` | `gemma4` |

### `verify_analysis`

对已有解析文本进行逐步骤校验。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `stem_text` | str | — | 原题文本 |
| `analysis` | str | — | 待校验的解析文本 |
| `verify_platform` | str | `"Qwen"` | 校验平台 |
| `verify_model` | str | `"qwen3-235b-a22b"` | 校验模型 |

返回每个步骤的 `verify_status`（`right` / `error` / `unknown`）及反馈。

### `ping`

检查后端解题/校验服务是否可达。

---

## 8. 解题流程（Skill）

`multi-model-math-solving` skill 定义了完整的协同解题流程：

1. **理解题目** — 确认题型、关键条件，文字化整理
2. **并行解题** — 在同一个 `function_calls` 块中同时调用 DeepSeek 和 Gemma 解题（禁止串行）
3. **交叉验证** — 用 Qwen（专职校验，不参与解题）对每个解答逐步骤验证
4. **迭代修正** — 若无候选正解，汇总错误与建议，综合改进后重新验证（最多 3 轮）
5. **输出答案** — 说明选择过程、各模型表现、修正了哪些关键错误

解题结束后直接返回答案，不写入记忆、不更新 Skill，也不执行额外的学习或反思流程。

### 已知陷阱

- **DeepSeek 辅助角公式易错**：处理 `acosθ + bsinθ` 时容易把 `Rcos(θ-φ)` 误写为 `Rsin(θ+φ)`，三角题中优先怀疑其辅助角步骤
- **模型分歧时优先数值验证**：用 Python 代入验证比再调一轮 LLM 更快更可靠

---

## 9. 批量解题

`test/test_batch_solve_xls.py` 支持从 Excel 文件批量读取题目并行解题。

```bash
python -m solver_agent.test.test_batch_solve_xls \
    --input problems.xlsx \
    --output-dir ./batch_results \
    --problem-col 题干文本 \
    --sheet 0 \
    --quiet
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input` | — | Excel 文件路径（`.xlsx` / `.xls`） |
| `--output-dir` | `./batch_results` | 输出目录 |
| `--problem-col` | `题干文本` | 题目所在列名 |
| `--sheet` | `0` | 工作表索引或名称 |
| `--quiet` | `false` | 抑制日志输出 |

输出结构：

```
batch_results/
├── md/          # 每题一个 Markdown（题干 + 解析）
│   ├── 0001.md
│   └── ...
└── json/        # 每题一个 JSON（完整 solve() 返回值）
    ├── 0001.json
    └── ...
```

---

## 10. Trace 全链路追踪

每次 `/solve` 请求自动在 `traces/` 下生成一份 JSON 文件（无论 `quiet` 是否为 `true`）。

文件名格式：`trc_<12位hex>.json`，与响应里的 `trace_id` 对应。

### Trace 文件结构

```json
{
  "trace_id": "trc_...",
  "created_at": "2026-05-01T12:00:00.000Z",
  "request": {
    "problem": "...",
    "quiet": false,
    "model": "gemma4",
    "provider": "custom",
    "api_mode": "chat_completions"
  },
  "result": {
    "answer": "...",
    "elapsed_seconds": 87.4,
    "completed": true,
    "api_calls": 5,
    "input_tokens": 2000,
    "output_tokens": 1500,
    "estimated_cost_usd": 0.05
  },
  "events": [
    {"type": "tool_start", "tool_call_id": "...", "tool_name": "...", "args": {}, "ts": "..."},
    {"type": "tool_complete", "tool_call_id": "...", "tool_name": "...", "result_chars": 500, "ts": "..."},
    {"type": "step", "api_call": 3, "prev_tools": [], "ts": "..."},
    {"type": "status", "level": "info", "message": "...", "ts": "..."}
  ],
  "messages": []
}
```

排查示例：
- 看 `events` 里的 `tool_start` / `tool_complete` 配对，确认双模型是否并行、各自耗时
- 看 `messages` 里 tool role 的 content，拿到每个模型的完整解析和验证结果
- 看 `events` 里的 `status` 事件，定位 ReadTimeout / 重连等异常

---

## 11. 验证流程是否生效

启动后随便发一题，**观察日志**应能看到：

| 信号 | 含义 |
|------|------|
| `Preloaded skills: ['multi-model-math-solving']` | skill 加载 OK |
| `MCP: registered N tool(s) from 1 server(s)` | MCP server 连上、工具注册成功 |
| 两个 `mcp_solver_solve_math_problem` 在同一回合并发触发 | skill 流程被严格遵循 |
| `verify_analysis` 调用，平台为 Qwen | 交叉验证按预期跑 |

如果只看到一个 `mcp_solver_solve_math_problem` 调用而不是两个，先确认请求里 `quiet=false`；若日志里仍只有一个 tool call，再考虑把 `model.default` 换强一点，或在题目前明确加一句"按 multi-model-math-solving 流程执行"。

---

## 12. 故障排查

| 症状 | 原因 / 处理 |
|------|------|
| 启动时报 `Failed to connect to MCP server 'solver'` | `command` 路径错 / 解释器没装 mcp 包 / 后端 `SOLVER_API_BASE` 不可达 |
| `Skill(s) not found: ['multi-model-math-solving']` | `solver_agent/skills/multi-model-math-solving/SKILL.md` 缺失，或 frontmatter `name:` 与目录名不一致 |
| 500 + `... unauthorized ...` | `custom_providers[0].api_key` 与模型端点不匹配 |
| Agent 串行而非并行调两模型 | 先确认日志里是否出现 2 条 `agent tool start: mcp_solver_solve_math_problem ...`；若只有 1 条，把 `agent.reasoning_effort` 提到 `high` |
| Compression model context is 65,434... | 压缩模型窗口略小于主模型压缩阈值；把 `compression.threshold` 从 `0.50` 调到 `0.49` |
| HTTP 客户端 504 / read timeout | 客户端超时 < 解题耗时；把客户端 timeout 拉到 600s+ |
| 多请求时第二个卡住 | MCP stdio 子进程通常单连接，多并发会排队 — 可调小 `ThreadPoolExecutor(max_workers=...)` 或换 HTTP transport |

---

## 13. 已知约束 / 后续扩展

- **单轮 + 同步**：当前接口不流式、不维护多轮上下文。如需，建议加 `POST /sessions` + `POST /sessions/{id}/turn` 走 SSE。
- **无鉴权**：内网部署直接用；若要外网暴露，加一个 `Depends(check_api_key)` 校验 `Authorization` 头即可。
- **配置热更新**：修改 `hermes_home/config.yaml` 后必须重启进程才能生效。
- **并发上限**：API 层 `ThreadPoolExecutor(max_workers=4)`，MCP stdio 子进程单连接排队，高并发场景需换 HTTP transport。
- **多 skill 串联**：`default_skills` 列表里追加更多 skill 即可被同时预加载。

---

## 14. 经验自动注入（与 evo_solver_agent 协同）

`solve_math_problem` MCP 工具在调用后端前会自动从共享知识库 `solver_agent/knowledge_base/`
检索相关历史经验，并追加到调用方传入的 `prompt` / `example` 字段（不覆盖）。

经验由姊妹 Agent **`evo_solver_agent`** 通过监督训练（题干 + 参考答案）持续沉淀：
解题→校验→反思→重试→提炼，无论对错都写入。

详见：
- `evo_solver_agent/README.md` — 训练 Agent 使用说明
- `solver_agent/knowledge_base/README.md` — 知识库格式与读写流程

控制开关：
| 变量 | 默认 | 作用 |
|------|------|------|
| `SOLVER_AUTO_EXPERIENCE` | `1` | 设为 `0` 关闭自动注入 |
| `SOLVER_KB_TOP_K`        | `3` | 注入条数上限 |
| `SOLVER_KB_LLM_RANK`     | `1` | 是否使用 LLM 对粗筛候选做语义重排 |
| `SOLVER_KB_DIR`          | `solver_agent/knowledge_base` | 知识库目录覆盖 |

新增 MCP 工具 `retrieve_experiences(problem, top_k)` 可由 Agent 显式查看
当前题目会注入哪些经验（不实际调用解题）。
