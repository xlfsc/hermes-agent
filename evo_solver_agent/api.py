"""FastAPI application for the evolutionary solver agent.

Endpoints:
    POST /train   — train on a single (problem, reference_answer) pair
    POST /batch   — train on a list of pairs
    GET  /health  — liveness probe
    GET  /stats   — knowledge base statistics
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evo_solver_agent.evo_solver import atrain

logger = logging.getLogger(__name__)

app = FastAPI(title="Evo Solver Agent", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    problem: str = Field(..., min_length=1, description="题干文本")
    reference_answer: str = Field(..., min_length=1, description="参考答案")


class TrainResponse(BaseModel):
    ok: bool
    correct: bool
    rounds_used: int
    elapsed_seconds: float
    experience_id: str
    final_answer: str
    reference_answer: str
    round_logs: List[Dict[str, Any]] = []


class BatchTrainRequest(BaseModel):
    items: List[TrainRequest] = Field(..., description="训练样本列表")
    max_workers: int = Field(2, ge=1, le=8)


class BatchTrainResponse(BaseModel):
    total: int
    correct_count: int
    accuracy: float
    results: List[TrainResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
async def train_endpoint(req: TrainRequest) -> TrainResponse:
    logger.info(
        "/train 收到请求 | 题干长度=%d | 参考答案长度=%d",
        len(req.problem),
        len(req.reference_answer)
    )
    try:
        result = await atrain(req.problem, req.reference_answer)
    except ValueError as exc:
        logger.warning("/train 请求参数错误: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("训练失败")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}"
        )
    logger.info(
        "/train 完成 | 是否正确=%s | 轮数=%d | 经验ID=%s",
        result.get("correct"),
        result.get("rounds_used", 0),
        result.get("experience_id", ""),
    )
    return TrainResponse(
        ok=result.get("ok", False),
        correct=result.get("correct", False),
        rounds_used=result.get("rounds_used", 0),
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
        experience_id=result.get("experience_id", ""),
        final_answer=result.get("final_answer", ""),
        reference_answer=result.get("reference_answer", ""),
        round_logs=result.get("round_logs", []),
    )


@app.post("/batch", response_model=BatchTrainResponse)
async def batch_train_endpoint(req: BatchTrainRequest) -> BatchTrainResponse:
    logger.info(
        "/batch 收到请求 | 样本数=%d | 最大并发=%d",
        len(req.items),
        req.max_workers
    )
    sem = asyncio.Semaphore(req.max_workers)

    async def _one(item: TrainRequest) -> Dict[str, Any]:
        async with sem:
            try:
                return await atrain(item.problem, item.reference_answer)
            except Exception as exc:
                logger.exception("批量训练单条失败")
                return {
                    "ok": False,
                    "correct": False,
                    "rounds_used": 0,
                    "elapsed_seconds": 0.0,
                    "experience_id": "",
                    "final_answer": "",
                    "reference_answer": item.reference_answer,
                    "round_logs": [],
                    "error": str(exc),
                }

    raw_results = await asyncio.gather(*[_one(item) for item in req.items])

    results: List[TrainResponse] = []
    correct_count = 0
    for r in raw_results:
        if r.get("correct"):
            correct_count += 1
        results.append(
            TrainResponse(
                ok=r.get("ok", False),
                correct=r.get("correct", False),
                rounds_used=r.get("rounds_used", 0),
                elapsed_seconds=r.get("elapsed_seconds", 0.0),
                experience_id=r.get("experience_id", ""),
                final_answer=r.get("final_answer", ""),
                reference_answer=r.get("reference_answer", ""),
                round_logs=r.get("round_logs", []),
            )
        )

    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    logger.info(
        "/batch 完成 | 总数=%d | 正确数=%d | 准确率=%.3f",
        total,
        correct_count,
        accuracy,
    )
    return BatchTrainResponse(
        total=total,
        correct_count=correct_count,
        accuracy=accuracy,
        results=results,
    )


@app.get("/stats")
def stats_endpoint() -> dict:
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    if str(_pkg_parent) not in _sys.path:
        _sys.path.insert(0, str(_pkg_parent))
    from solver_agent.knowledge import load_experiences
    exps = load_experiences()
    correct = sum(1 for e in exps if e.final_correct)
    return {
        "total_experiences": len(exps),
        "correct_count": correct,
        "incorrect_count": len(exps) - correct,
        "accuracy": correct / len(exps) if exps else 0.0,
    }
