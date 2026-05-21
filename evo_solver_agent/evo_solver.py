"""Evolutionary solver agent — the training loop.

Inputs:  problem text + reference answer (supervised signal)
Outputs: a verdict (correct / incorrect), the rounds it took, the rendered
         best solution, and an Experience record that has been persisted to
         the shared knowledge base for later use by ``solver_agent``.

Per-round flow::

    retrieve → solve(DeepSeek) || solve(Gemma) → verify_each → compare_to_ref
        ↓ if any attempt passes both step-verification AND ref-comparison ↓
        success — distill experience and persist
        ↓ else
        reflect → next round (max 3 rounds)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from solver_agent.backend_client import BackendClient
from solver_agent.knowledge import (
    add_experience,
    fetch_injection,
    merge_prompt,
)
from solver_agent.llm_client import LLMClient

from evo_solver_agent.learner import Learner

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = int(os.environ.get("EVO_MAX_ROUNDS", "3"))
DEFAULT_KB_TOP_K = int(os.environ.get("EVO_KB_TOP_K", "3"))
DEFAULT_USE_LLM_RANK = os.environ.get("EVO_KB_LLM_RANK", "1").lower() not in {
    "0", "false", "no", ""}

# Solver platform/model pairs used in parallel each round.
SOLVER_LINEUP: List[Tuple[str, str]] = [
    ("DeepSeek", "deepseek-v3.2"),
    ("Gemma", "gemma4"),
]


class EvoSolver:
    """Trainer that learns from supervised (problem, reference_answer) pairs."""

    def __init__(
            self,
            backend: Optional[BackendClient] = None,
            llm: Optional[LLMClient] = None,
            learner: Optional[Learner] = None,
            max_rounds: int = DEFAULT_MAX_ROUNDS,
            kb_top_k: int = DEFAULT_KB_TOP_K,
            use_llm_rank: bool = DEFAULT_USE_LLM_RANK,
    ) -> None:
        self.backend = backend or BackendClient()
        self.llm = llm or LLMClient()
        self.learner = learner or Learner(self.llm)
        self.max_rounds = max(1, int(max_rounds))
        self.kb_top_k = kb_top_k
        self.use_llm_rank = use_llm_rank

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def train(self, problem: str, reference_answer: str, *,
              persist: bool = True) -> Dict[str, Any]:
        """Sync wrapper around :meth:`atrain` for callers without an event loop."""
        return asyncio.run(
            self.atrain(problem, reference_answer, persist=persist))

    async def atrain(
            self, problem: str, reference_answer: str, *, persist: bool = True
    ) -> Dict[str, Any]:
        """Run the full training loop on a single supervised example."""

        if not problem or not problem.strip():
            raise ValueError("problem must be non-empty")
        if not reference_answer or not reference_answer.strip():
            raise ValueError("reference_answer must be non-empty")

        t0 = time.time()
        all_attempts: List[Dict[str, Any]] = []
        all_verifications: List[Dict[str, Any]] = []
        round_logs: List[Dict[str, Any]] = []

        logger.info(
            "训练开始 | 最大轮数=%d | 题干=%s | 参考答案=%s",
            self.max_rounds,
            problem.strip()[:80].replace("\n", " "),
            reference_answer.strip()[:80].replace("\n", " "),
        )

        # 1) one-shot retrieval (per user's directive: retrieve once, inject twice)
        try:
            kb_inj = fetch_injection(
                problem,
                top_k=self.kb_top_k,
                use_llm=self.use_llm_rank
            )
            logger.info(
                "知识库检索成功 | 命中数=%d | 命中ID=%s",
                kb_inj.get("matched_count", 0),
                kb_inj.get("matched_ids", []),
            )
        except Exception as exc:
            logger.warning("知识库检索失败: %s", exc)
            kb_inj = {
                "prompt": "",
                "example": "",
                "matched_ids": [],
                "matched_count": 0
            }

        base_prompt = kb_inj.get("prompt", "")
        base_example = kb_inj.get("example", "")
        reflection_text = ""

        winner: Optional[Dict[str, Any]] = None
        rounds_used = 0

        for round_idx in range(1, self.max_rounds + 1):
            rounds_used = round_idx
            logger.info(
                "第 %d/%d 轮开始 | 是否使用反思=%s",
                round_idx,
                self.max_rounds,
                bool(reflection_text),
            )

            round_prompt = merge_prompt(base_prompt, reflection_text)
            attempts = await self._solve_parallel(
                problem,
                prompt=round_prompt,
                example=base_example
            )
            all_attempts.extend(attempts)

            verifications = await self._verify_and_compare_parallel(
                problem, reference_answer, attempts, round_idx
            )
            all_verifications.extend(verifications)

            round_logs.append({
                "round": round_idx,
                "reflection_used": bool(reflection_text),
                "reflection_text": reflection_text,
                "attempts": [
                    {
                        "platform": a.get("platform"),
                        "model": a.get("model"),
                        "ok": a.get("ok", False),
                        "correct": a.get("correct", False),
                        "analysis": a.get("final_analysis") or a.get("analysis") or "",
                        "verify": a.get("verify") or {},
                        "compare": a.get("compare") or {},
                        "error": a.get("error"),
                    }
                    for a in attempts
                ],
            })

            winner = next((a for a in attempts if a.get("correct")), None)
            if winner is not None:
                logger.info(
                    "第 %d 轮 | 胜出模型=%s — 退出循环",
                    round_idx,
                    winner.get("model"),
                )
                break

            # No correct attempt — generate a reflection for the next round.
            if round_idx < self.max_rounds:
                logger.info(
                    "第 %d 轮 | 无正确答案，生成下一轮反思",
                    round_idx
                )
                reflection_text = self.learner.reflect(
                    problem,
                    reference_answer,
                    attempts,
                    verifications,
                )
                logger.info(
                    "第 %d 轮 | 反思已生成（%d 字符）",
                    round_idx,
                    len(reflection_text or ""),
                )

        final_correct = winner is not None
        best = winner or self._pick_least_bad(all_attempts)
        best_solution = (best.get("final_analysis") or best.get(
            "analysis") or "") if best else ""
        final_answer = self._extract_final_answer(best_solution) if best else ""

        # 2) distill into a reusable Experience and persist
        experience = self.learner.distill(
            problem=problem,
            reference_answer=reference_answer,
            attempts=all_attempts,
            verifications=all_verifications,
            final_correct=final_correct,
            rounds_used=rounds_used,
            best_solution=best_solution if final_correct else "",
            final_answer=final_answer,
        )
        if persist:
            add_experience(experience)
            logger.info("经验已持久化 | id=%s", experience.id)

        elapsed = time.time() - t0
        logger.info(
            "训练结束 | 是否正确=%s | 使用轮数=%d | 总耗时=%.2fs | 经验ID=%s",
            final_correct,
            rounds_used,
            elapsed,
            experience.id,
        )
        return {
            "ok": True,
            "correct": final_correct,
            "rounds_used": rounds_used,
            "elapsed_seconds": elapsed,
            "experience_id": experience.id,
            "final_answer": final_answer,
            "best_solution": best_solution,
            "reference_answer": reference_answer,
            "kb_used": {
                "matched_ids": kb_inj.get("matched_ids", []),
                "matched_count": kb_inj.get("matched_count", 0),
            },
            "round_logs": round_logs,
            "experience": experience.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _solve_parallel(
            self, problem: str, *, prompt: str, example: str
    ) -> List[Dict[str, Any]]:
        """Call DeepSeek + Gemma concurrently, one round."""

        async def _one(platform: str, model: str) -> Dict[str, Any]:
            t_start = time.time()
            logger.info("解题开始 | 平台=%s | 模型=%s", platform, model)
            try:
                resp = await self.backend.solve_problem(
                    text_input=problem,
                    solve_platform=platform,
                    solve_model=model,
                    thinking=False,
                    verify=False,
                    prompt=prompt,
                    example=example,
                )
            except Exception as exc:
                logger.exception(
                    "解题失败 | 平台=%s | 模型=%s", platform, model
                )
                resp = {"ok": False, "error": str(exc)}
            resp = dict(resp) if isinstance(resp, dict) \
                else {"ok": False, "raw": resp}

            resp["model"] = model
            resp["platform"] = platform
            logger.info(
                "解题完成 | 平台=%s | 模型=%s | 是否成功=%s | 耗时=%.2fs",
                platform,
                model,
                resp.get("ok", False),
                time.time() - t_start,
            )
            return resp

        return list(
            await asyncio.gather(*[_one(p, m) for p, m in SOLVER_LINEUP])
        )

    async def _verify_and_compare_parallel(
            self,
            problem: str,
            reference_answer: str,
            attempts: List[Dict[str, Any]],
            round_idx: int,
    ) -> List[Dict[str, Any]]:
        """For each attempt, run backend verify + LLM compare concurrently.

        ``verify_analysis`` is async I/O; ``compare_answers`` is a blocking LLM
        call, so it is dispatched to a worker thread via ``asyncio.to_thread``.
        Both run in parallel within an attempt, and all attempts are gathered
        across the round.
        """

        async def _one(att: Dict[str, Any]) -> Dict[str, Any]:
            model = att.get("model", "?")
            analysis = att.get("final_analysis") or att.get("analysis") or ""
            if not analysis:
                logger.warning(
                    "第 %d 轮 | 模型=%s | 解析为空", round_idx, model
                )
                att["correct"] = False
                att["verify"] = {"ok": False, "error": "empty_analysis"}
                return {"verify_results": []}

            verify_task = asyncio.create_task(
                self.backend.verify_analysis(problem, analysis)
            )
            compare_task = asyncio.create_task(
                asyncio.to_thread(
                    self.learner.compare_answers,
                    problem,
                    analysis,
                    reference_answer
                )
            )
            verify, cmp = await asyncio.gather(verify_task, compare_task)

            att["verify"] = verify
            att["compare"] = cmp
            step_ok = verify.get("ok") and not verify.get("has_error", True)
            att["correct"] = bool(step_ok and cmp.get("correct", False))

            logger.info(
                "第 %d 轮 | 模型=%s | 步骤校验通过=%s | 答案匹配=%s | 整体正确=%s",
                round_idx,
                model,
                step_ok,
                cmp.get("correct", False),
                att["correct"],
            )
            return verify

        return list(await asyncio.gather(*[_one(att) for att in attempts]))

    @staticmethod
    def _pick_least_bad(attempts: List[Dict[str, Any]]) -> Optional[
        Dict[str, Any]]:
        if not attempts:
            return None
        # Prefer attempts where step verification passed even if ref comparison failed.
        for a in attempts:
            v = a.get("verify") or {}
            if v.get("ok") and not v.get("has_error", True):
                return a
        # Otherwise pick the most recent non-empty analysis.
        for a in reversed(attempts):
            if a.get("final_analysis") or a.get("analysis"):
                return a
        return attempts[-1]

    @staticmethod
    def _extract_final_answer(solution: str) -> str:
        """Best-effort scrape of a final-answer line."""
        if not solution:
            return ""
        markers = ["最终答案", "答案:", "答案：", "Answer:", "因此,", "因此，", "综上,", "综上，"]
        for m in markers:
            idx = solution.rfind(m)
            if idx != -1:
                tail = solution[idx: idx + 200].strip()
                return tail
        # Fallback: last non-empty line, trimmed
        lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
        return lines[-1][:200] if lines else ""


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


_default_evo: Optional[EvoSolver] = None


def get_default_evo() -> EvoSolver:
    global _default_evo
    if _default_evo is None:
        _default_evo = EvoSolver()
    return _default_evo


def train(
        problem: str, reference_answer: str, *, persist: bool = True
) -> Dict[str, Any]:
    return get_default_evo().train(
        problem, reference_answer, persist=persist
    )


async def atrain(
        problem: str, reference_answer: str, *, persist: bool = True
) -> Dict[str, Any]:
    return await get_default_evo().atrain(
        problem,
        reference_answer,
        persist=persist
    )
