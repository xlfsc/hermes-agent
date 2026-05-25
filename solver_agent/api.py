"""FastAPI 应用：POST /solve {problem} -> {answer, elapsed_seconds, model}"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from solver_agent.solver import solve

logger = logging.getLogger(__name__)

app = FastAPI(title="Math Solver Agent", version="0.1.0")
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="solver")


class SolveRequest(BaseModel):
    problem: str = Field(..., min_length=1, description="题目文本")
    quiet: bool = Field(False, description="是否抑制 Agent stdout/stderr")


class SolveResponse(BaseModel):
    answer: str
    elapsed_seconds: float
    model: str
    trace_id: str


@app.get("/health")
def health() -> dict:
    logger.info("健康检查请求 | 端点=/health")
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
async def solve_endpoint(req: SolveRequest) -> SolveResponse:
    logger.info(
        "收到 /solve 请求 | 题干长度=%d | quiet=%s",
        len(req.problem),
        req.quiet,
    )
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _pool, lambda: solve(req.problem, quiet=req.quiet)
        )
    except ValueError as exc:
        logger.warning("/solve 入参非法 | 错误=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("/solve 处理失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    logger.info(
        "/solve 处理完成 | trace_id=%s | model=%s | 耗时=%.3fs | 答案长度=%d",
        result.get("trace_id"),
        result.get("model"),
        result.get("elapsed_seconds", 0.0),
        len(result.get("answer") or ""),
    )
    return SolveResponse(**result)
