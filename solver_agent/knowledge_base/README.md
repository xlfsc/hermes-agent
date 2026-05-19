# knowledge_base

`solver_agent` 与 `evo_solver_agent` 共享的经验持久化目录。

## 文件

- `experiences.jsonl` — append-only，每行一条完整 `Experience`
- `signatures.jsonl`  — append-only，每行一条紧凑摘要（用于检索时的 LLM 排序）

## 写入方

`evo_solver_agent.evo_solver.EvoSolver.train()` 在每条监督样本训练完成后，
无论对错都会调用 `solver_agent.knowledge.add_experience()` 追加一行。

## 读取方

`solver_agent.solver_mcp_server.solve_math_problem` 在调用后端 `/api/solve_problem`
之前，会自动检索相关经验并注入到 `prompt` / `example`：

1. **关键词倒排粗筛**（基于题型 + 关键概念，候选 ≤ 12 条）
2. **LLM 重排**（gemma4 默认）取 Top-K (默认 3 条)
3. 渲染为 prompt（关键思路 + 易错点）+ example（最相关的正确解题示范）
4. 与调用方传入的 prompt/example **追加合并**，不覆盖

## 关闭注入

设置环境变量 `SOLVER_AUTO_EXPERIENCE=0` 即可关闭自动注入。

## 调整位置

设置 `SOLVER_KB_DIR=/path/to/your/dir` 可使用其他目录（多环境/多租户场景）。

## 数据格式

每行 `experiences.jsonl`：
```json
{
  "id": "exp_<12hex>",
  "ts": "2026-05-19T22:30:00",
  "problem": "...",
  "problem_type": "三角函数",
  "keywords": ["sin", "cos", "辅助角"],
  "reference_answer": "-3/4",
  "final_correct": true,
  "rounds_used": 1,
  "key_insights": ["平方两边可消去交叉项..."],
  "error_patterns": [
    {"model": "deepseek-v3.2", "pattern": "辅助角公式系数错误", "correction": "..."}
  ],
  "best_solution": "...",
  "final_answer": "-3/4",
  "model_observations": {"deepseek-v3.2": "首轮答错", "gemma4": "首轮答对"}
}
```

## 备份与归档

JSONL 是 append-only 文本格式，可直接用 git 跟踪或 `tar` 备份。
如需清理旧数据，可定期归档为 `experiences-YYYYMM.jsonl` 后重建主文件。
