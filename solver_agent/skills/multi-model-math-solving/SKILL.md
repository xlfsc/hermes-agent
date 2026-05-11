---
name: multi-model-math-solving
description: 自动并行调用不同大模型解答数学题，并利用逐步骤验证与交叉校验实现自我纠错，持续优化策略
---

# 多模型协同解题与验证流程

## 触发条件
当用户提供一道数学题文本，并要求解题时，自动执行本流程。

## 步骤

### 第一步：理解题目
- 确认题目类型（代数、几何、微积分等）和关键条件。
- 如果题目包含图片或复杂公式，先做文字化整理。

### 第二步：并行调用不同模型解题
- **必须在同一个 function_calls 块中同时发起两个 `mcp_solver_solve_math_problem` 调用**，实现真正的并行执行，禁止串行逐个调用。
- 两个调用分别为：
  1. `mcp_solver_solve_math_problem(text_input=..., solve_platform="DeepSeek", solve_model="deepseek-v3.2", thinking=true)`
  2. `mcp_solver_solve_math_problem(text_input=..., solve_platform="Gemma", solve_model="gemma4", thinking=true)`
- **禁止使用 delegate_task 串行调用**，直接在一个回合内并发发出两个 MCP 工具调用即可。
- 等待所有调用返回后，进入第三步。

- 工具名称：`mcp_solver_solve_math_problem`
- 参数说明：
  - `text_input`：题目文本
  - `solve_platform`：平台名，可选 `"DeepSeek"`、`"Gemma"`
  - `solve_model`：模型名，必须与平台匹配：
    - `DeepSeek` → `deepseek-v3.2`
    - `Gemma` → `gemma4`
  - `thinking`：`true` 或 `false`，是否开启深度思考
  - `prompt`：自定义解题提示词，可空使用服务端默认
- **关键：两个调用必须放在同一个 function_calls 块中，利用底层并行执行，大幅缩短总耗时。**

### 第三步：双路并行验证（Qwen + Lean4，取并集）
- 收集所有模型的解答后，对每个答案 **在同一个 function_calls 块中并行调用两个验证工具**：
  1. `mcp_solver_verify_analysis(stem_text=..., analysis=..., verify_platform="Qwen", verify_model="qwen3-235b-a22b")` — 基于 LLM 的自然语言逐步骤校验。
  2. `mcp_solver_verify_analysis_lean4(problem_text=..., solution_text=...)` — 基于 Lean4 的形式化校验，机器可证明级别。
- 两个工具是 **独立的验证路径**，禁止串行；同一回合内并发发出即可。
- **`verify_analysis`** 的返回（Qwen 路）：
  - 对每个步骤标注：`正确`、`错误` 或 `未知`
  - 若为 `错误`，附带 `错因`、`建议修正点`
  - 汇总各类计数与整体置信度
- **`verify_analysis_lean4`** 的返回（Lean4 路）：
  - `success`：本次形式化校验是否整体通过（等价于 `not has_error`）
  - `stage`：失败所在阶段 `formalization` | `compile` | `semantic` | `system`
  - `formalization.lean_code`：转换出的 Lean4 代码（通过的话可作为正确性证据）
  - `compile_check.compile_pass`、`semantic_check.semantic_pass`：两阶段的布尔结果
  - `conclusion.first_error`、`conclusion.message`：首条关键错误与总结
- **结果取并集（联合判定）**：
  - 只有当两路都判为通过时，该解答才算"形式上正确"
  - 任一路报错即视为该步骤存疑，进入"待修正"列表
  - Lean4 在 `formalization` 或 `system` 阶段失败时视为"无法形式化"，不计入错误；以 Qwen 的判定为准
  - 只有所有步骤都通过两路校验时，该解答才标记为"候选正解"
- **交叉验证要求**：
  - 每个解答都要走完整的双路验证
  - 若仅有一个候选正解，直接进入第五步输出
  - 否则进入第四步迭代修正


### 第四步：迭代修正直到产生正解
如果当前轮次没有找到任何"候选正解"（即所有解答均存在错误步骤），则执行以下迭代逻辑：

1. **汇总错误与建议**：收集所有 `verify_analysis` 返回的错误步骤、错因及修正建议，合并去重。
2. **生成综合改进解答**：
   - 综合不同模型解答中的正确步骤片段
   - 根据修正建议修正错误步骤
   - 若某些步骤所有模型均无法确定或矛盾，则将其标记为"待推理"，并尝试用更强的验证模型（如 Qwen 的 reasoning 模式）再次分析
3. **重新调用解题工具（可选）**：
   - 使用上一轮中表现最好的模型，结合汇总的修正提示，以更明确的 `prompt`（例如"请重点检查第 X 步的等价变换，避免出现 XX 类错误"）再次调用 `solve_math_problem`
   - 或直接使用综合改进后的解答作为新的答案，并再次调用 `verify_analysis` 验证
4. **再次验证**：对新生成的解答调用 `verify_analysis`，检查是否所有步骤均为正确。
5. **终止条件**：
   - 找到候选正解（所有步骤正确）→ 停止迭代，输出该解
   - 达到最大迭代次数（建议 3 轮）仍无正解 → 输出当前最接近正确的解答，并明确标出仍然存疑的步骤，供人工判断

### 第五步：输出最终答案与过程解释
- 输出最终确定的正确解析（或最优解）。
- 简要说明选择过程：
  - 列出各模型的初始正确/错误步骤数
  - 说明迭代修正了哪些关键错误
  - 如果最终为综合解答，说明各部分来源与融合依据

### 第六步：沉淀经验（自我进化）
- 在会话结束前，用 `/memory add` 记录：
  - 题目类型、错误模式（如"对数换底时常量错位"）
  - 高成功率模型组合（如"三角题 Qwen+Gemma 互补性强"）
- 完成反思提示后，执行：
  - `/compress` 清理上下文
  - 询问："本次解题流程有哪些可优化之处？"并让 Agent 更新本 Skill 文件或记忆。

## 常见陷阱与经验

### 模型分歧时优先用数值验证
当模型在某个选项上产生分歧（如 2:1），不要只依赖 `verify_analysis`（它对辅助角公式等变换的校验能力有限）。用 `execute_code` 做 Python 数值代入验证，几行代码就能确认哪个模型对：
```python
import math
# 直接代入已知值验证等式是否成立
val = cos2a * math.cos(3*math.pi/4) - sin2a * math.sin(3*math.pi/4)
print(abs(val - target) < 1e-10)  # True/False 一目了然
```
这比再调一轮 LLM 验证更快、更可靠。

### DeepSeek 辅助角公式易错
DeepSeek 在处理 `acosθ + bsinθ` 的辅助角变换时，容易把 `Rcos(θ-φ)` 误写为 `Rsin(θ+φ)`，导致后续数值计算全错。三角题中如果 DeepSeek 与其他模型分歧，优先怀疑其辅助角步骤。

### Lean4 对几何/应用题常返回 unknown
`verify_analysis_lean4` 擅长代数恒等式、方程推导、数值不等式等可形式化的题型；
对纯几何、文字应用题、需要外部公式库的题型，常在 `stage="formalization"` 阶段
直接失败（自然语言难以自动转成 Lean4 代码）——这不代表解答错误，只代表
"无法形式化"。此时以 `verify_analysis` 的 Qwen 结果为准即可。

## 可调参数
- 初始并行解题平台列表：`["DeepSeek", "Gemma"]`
- 验证路径（双路并行，取并集）：
  - LLM 路：`verify_analysis`，平台 `Qwen`（`qwen3-235b-a22b`，专职校验，不参与解题）
  - 形式化路：`verify_analysis_lean4`（Lean4 后端，无需指定模型）
- 平台与模型组合（解题）：
  - `DeepSeek` → `deepseek-v3.2`
  - `Gemma` → `gemma4`
- 最大迭代轮数：`3`
- 最大并发工具调用数：`5`
