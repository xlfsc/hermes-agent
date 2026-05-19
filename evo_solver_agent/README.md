# evo_solver_agent

自我推理解题、自我验证、自我学习和进化的 Agent。

输入 `(题干, 参考答案)` 监督样本，自动完成：
1. 检索历史经验 → 注入到解题 prompt
2. DeepSeek + Gemma 并行解题
3. Qwen 步骤校验 + LLM 语义比对参考答案
4. 若不正确则反思 → 注入 → 重试（最多 3 轮）
5. 提炼经验写入共享知识库 `solver_agent/knowledge_base/`

学到的经验会被 `solver_agent` 透明复用：每次调用 `solve_math_problem` 时
MCP 服务端会自动检索 Top-K 相关经验并注入到 `prompt`/`example`。

## 目录结构

```
evo_solver_agent/
├── evo_solver.py        # 训练循环 EvoSolver / train(problem, ref_answer)
├── learner.py           # 反思 + 答案比对 + 经验提炼（基于 LLM）
├── api.py               # FastAPI: POST /train, POST /batch, GET /stats
├── run_server.py        # uvicorn 启动入口
└── test/
    └── test_batch_train_xls.py   # Excel 批量训练
```

依赖共享模块（位于 `solver_agent/`）：

- `solver_agent/backend_client.py` — solve / verify HTTP 客户端
- `solver_agent/llm_client.py`     — OpenAI 兼容 chat 客户端
- `solver_agent/knowledge.py`      — 经验读写 / 检索 / 注入文本生成
- `solver_agent/knowledge_base/`   — 经验持久化目录（JSONL）

## 启动

```bash
python evo_solver_agent/run_server.py     # 默认 0.0.0.0:8766
```

## API

### POST /train

```json
{
  "problem": "已知 sin x + cos x = 1/2，求 sin 2x 的值。",
  "reference_answer": "-3/4"
}
```

返回：
```json
{
  "ok": true,
  "correct": true,
  "rounds_used": 1,
  "elapsed_seconds": 28.4,
  "experience_id": "exp_a1b2c3d4e5f6",
  "final_answer": "...",
  "reference_answer": "-3/4",
  "round_logs": [...]
}
```

### POST /batch

```json
{
  "items": [{"problem": "...", "reference_answer": "..."}, ...],
  "max_workers": 2
}
```

返回 batch summary 与每条结果。

### GET /stats

```json
{"total_experiences": 142, "correct_count": 118, "incorrect_count": 24, "accuracy": 0.83}
```

## 批量训练（Excel）

Excel 至少包含两列：题干、参考答案。

```bash
python -m evo_solver_agent.test.test_batch_train_xls \
  --input ./monitored.xlsx \
  --problem-col 题干文本 \
  --answer-col 参考答案 \
  --output-dir ./train_results \
  --max-workers 2
```

输出：
- `train_results/json/<seq>.json` — 每题完整训练结果
- `train_results/md/<seq>.md`     — 题目 + 参考答案 + Agent 答案 markdown
- `train_results/summary.json`    — 整体准确率

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EVO_HOST`           | `0.0.0.0`                   | 服务监听地址 |
| `EVO_PORT`           | `8766`                       | 服务端口 |
| `EVO_MAX_ROUNDS`     | `3`                          | 单题最大重试轮数 |
| `EVO_KB_TOP_K`       | `3`                          | 注入经验条数 |
| `EVO_KB_LLM_RANK`    | `1`                          | 是否启用 LLM 排序 |
| `SOLVER_API_BASE`    | `http://172.168.80.46:8000`  | 解题/校验后端 |
| `SOLVER_KB_DIR`      | `solver_agent/knowledge_base` | 知识库目录 |
| `KB_LLM_BASE_URL`    | `http://171.214.10.150:11600/v1/` | 反思/检索 LLM |
| `KB_LLM_API_KEY`     | `mysecurekey123`             | 反思/检索 LLM API key |
| `KB_LLM_MODEL`       | `gemma4`                      | 反思/检索 LLM 模型 |

## 经验沉淀机制

每条训练完成后，无论对错都会持久化为一条 `Experience` 写入：

- `knowledge_base/experiences.jsonl` — 完整记录
- `knowledge_base/signatures.jsonl`  — 紧凑摘要（用于检索）

`Experience` 字段：
- `id`, `ts`, `problem`, `problem_type`, `keywords`
- `reference_answer`, `final_correct`, `rounds_used`
- `key_insights[]` — 本题成功的关键思路（即使答错也会保留尝试中的可取之处）
- `error_patterns[]` — 哪个模型犯了什么错、应当如何修正
- `model_observations{}` — 各模型在该题上的表现描述
- `best_solution`, `final_answer`

## 训练→上线 通路

```
evo_solver_agent.train(problem, ref_answer)
    ↓ 写入
solver_agent/knowledge_base/{experiences,signatures}.jsonl
    ↓ MCP 服务端启动时透明读取
solver_agent.solver_mcp_server.solve_math_problem()
    ↓ 调用前自动检索并追加到 prompt/example
后端 /api/solve_problem
```

无需重启 MCP 服务即可生效（每次调用前重新加载 JSONL，文件 mtime 变化时
缓存自动失效）。
