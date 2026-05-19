"""Learner module — reflection and experience distillation.

Given a problem, its reference answer, and the solving attempts (with
verification results), this module:

1. **reflect()** — Analyses why the attempt failed and produces a concise
   reflection that can be injected into the next retry round.
2. **distill()** — After the training loop finishes (success or max-rounds),
   extracts a reusable Experience record for the shared knowledge base.
3. **compare_answers()** — LLM-based semantic comparison between a candidate
   answer and the reference answer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from solver_agent.llm_client import LLMClient
from solver_agent.knowledge import (
    Experience,
    ErrorPattern,
    extract_keywords,
    infer_problem_type,
    make_experience_id,
    now_iso,
)


class Learner:
    """Stateless helper — all state lives in the caller (evo_solver)."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Answer comparison
    # ------------------------------------------------------------------

    def compare_answers(self, problem: str, candidate: str, reference: str) -> Dict[str, Any]:
        """Semantic comparison: is the candidate answer equivalent to reference?

        Returns {"correct": bool, "reason": str}.
        """
        system = (
            "你是数学答案判定专家。给定一道数学题、一个候选答案和一个参考答案，"
            "判断候选答案是否与参考答案等价（允许不同表达形式，如 0.5 = 1/2 = 二分之一）。"
            "严格输出 JSON: {\"correct\": true/false, \"reason\": \"简短理由\"}"
        )
        user = (
            f"题目:\n{problem.strip()[:1500]}\n\n"
            f"候选答案:\n{candidate.strip()[:800]}\n\n"
            f"参考答案:\n{reference.strip()[:800]}"
        )
        try:
            data = self.llm.chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=300,
            )
            return {
                "correct": bool(data.get("correct", False)),
                "reason": str(data.get("reason", "")),
            }
        except Exception as exc:
            return {"correct": False, "reason": f"comparison_error: {exc}"}

    # ------------------------------------------------------------------
    # Reflection (for retry injection)
    # ------------------------------------------------------------------

    def reflect(
        self,
        problem: str,
        reference_answer: str,
        attempts: List[Dict[str, Any]],
        verifications: List[Dict[str, Any]],
    ) -> str:
        """Produce a concise reflection to inject into the next solving round.

        The reflection should:
        - Identify the root cause of failure (conceptual error, calculation, etc.)
        - Suggest a corrected approach
        - Be short enough to fit in a prompt injection (~500 chars)
        """
        system = (
            "你是数学解题教练。学生做错了一道题，你需要分析错因并给出简短的修正指引，"
            "帮助学生在下一次尝试中避免同样的错误。输出纯文本，不超过 400 字。"
        )

        attempts_summary = ""
        for idx, att in enumerate(attempts[-2:], 1):
            model = att.get("model", "?")
            answer = (att.get("final_analysis") or att.get("analysis") or "")[:600]
            attempts_summary += f"\n--- 尝试 {idx} ({model}) ---\n{answer}\n"

        verify_summary = ""
        for v in verifications[-2:]:
            results = v.get("verify_results", [])
            errors = [r for r in results if r.get("verify_status") == "error"]
            if errors:
                for e in errors[:3]:
                    verify_summary += f"- 步骤{e.get('step_idx','?')}: {e.get('feedback_content','')[:200]}\n"

        user = (
            f"题目:\n{problem.strip()[:1200]}\n\n"
            f"参考答案:\n{reference_answer.strip()[:600]}\n\n"
            f"学生的解题尝试:{attempts_summary}\n\n"
            f"校验发现的错误:\n{verify_summary}\n\n"
            "请分析错因并给出修正指引（简短、可操作）。"
        )

        try:
            return self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=600,
            )
        except Exception as exc:
            return f"[反思生成失败: {exc}]"

    # ------------------------------------------------------------------
    # Distillation (for knowledge base)
    # ------------------------------------------------------------------

    def distill(
        self,
        problem: str,
        reference_answer: str,
        attempts: List[Dict[str, Any]],
        verifications: List[Dict[str, Any]],
        final_correct: bool,
        rounds_used: int,
        best_solution: str = "",
        final_answer: str = "",
    ) -> Experience:
        """Distill the training episode into a reusable Experience record."""

        system = (
            "你是数学解题经验总结专家。根据一道题的解题过程（可能经历了多轮尝试），"
            "提炼出可复用的经验。严格输出 JSON:\n"
            "{\n"
            "  \"key_insights\": [\"洞察1\", \"洞察2\"],\n"
            "  \"error_patterns\": [{\"model\": \"模型名\", \"pattern\": \"错误模式\", \"correction\": \"修正方法\"}],\n"
            "  \"model_observations\": {\"模型名\": \"表现描述\"}\n"
            "}"
        )

        attempts_summary = ""
        for idx, att in enumerate(attempts[-4:], 1):
            model = att.get("model", "?")
            correct = att.get("correct", "?")
            answer = (att.get("final_analysis") or att.get("analysis") or "")[:400]
            attempts_summary += f"\n[尝试{idx}] model={model} correct={correct}\n{answer}\n"

        verify_summary = ""
        for v in verifications[-4:]:
            results = v.get("verify_results", [])
            errors = [r for r in results if r.get("verify_status") == "error"]
            for e in errors[:2]:
                verify_summary += f"- 步骤{e.get('step_idx','?')}: {e.get('feedback_content','')[:150]}\n"

        user = (
            f"题目:\n{problem.strip()[:1200]}\n\n"
            f"参考答案: {reference_answer.strip()[:400]}\n\n"
            f"最终是否正确: {'是' if final_correct else '否'}\n"
            f"使用轮数: {rounds_used}\n\n"
            f"各轮尝试:{attempts_summary}\n\n"
            f"校验错误:\n{verify_summary}\n\n"
            "请提炼经验。"
        )

        try:
            data = self.llm.chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=800,
            )
        except Exception:
            data = {}

        key_insights = data.get("key_insights", []) if isinstance(data, dict) else []
        raw_eps = data.get("error_patterns", []) if isinstance(data, dict) else []
        model_obs = data.get("model_observations", {}) if isinstance(data, dict) else {}

        error_patterns = []
        for ep in raw_eps:
            if isinstance(ep, dict):
                error_patterns.append(ErrorPattern(
                    model=ep.get("model", ""),
                    pattern=ep.get("pattern", ""),
                    correction=ep.get("correction", ""),
                ))

        ts = now_iso()
        exp_id = make_experience_id(problem, ts)

        return Experience(
            id=exp_id,
            ts=ts,
            problem=problem,
            problem_type=infer_problem_type(problem),
            keywords=extract_keywords(problem),
            reference_answer=reference_answer,
            final_correct=final_correct,
            rounds_used=rounds_used,
            key_insights=list(key_insights) if isinstance(key_insights, list) else [],
            error_patterns=error_patterns,
            best_solution=best_solution,
            final_answer=final_answer,
            model_observations=dict(model_obs) if isinstance(model_obs, dict) else {},
        )
