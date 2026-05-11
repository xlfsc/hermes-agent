"""
Solver MCP Server.

Wraps the in-house FastAPI solving / verification service so that Hermes
Agent (and any other MCP client) can invoke it as native tools.

Transport: stdio (default for FastMCP).
Backend  : http://172.168.80.46:8000  (override via SOLVER_API_BASE env var)

Tools exposed:
  - solve_math_problem
  - verify_analysis
  - ping
"""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP


API_BASE = os.environ.get("SOLVER_API_BASE", "http://172.168.80.46:8000").rstrip("/")
API_KEY = os.environ.get("SOLVER_API_KEY")

SOLVE_TIMEOUT = float(os.environ.get("SOLVER_SOLVE_TIMEOUT", "600"))
VERIFY_TIMEOUT = float(os.environ.get("SOLVER_VERIFY_TIMEOUT", "300"))

LEAN4_API_BASE = os.environ.get(
    "LEAN4_API_BASE", "http://172.168.80.36:8008"
).rstrip("/")
LEAN4_VERIFY_TIMEOUT = float(os.environ.get("LEAN4_VERIFY_TIMEOUT", "300"))

PLATFORM_MODELS = {
    "Qwen":     ["qwen3-235b-a22b"],
    "DeepSeek": ["deepseek-v3.2"],
    "Gemma":    ["gemma4"],
}

mcp = FastMCP("solver")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _truncate(text: Any, limit: int = 4000) -> Any:
    if not isinstance(text, str):
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


@mcp.tool()
async def solve_math_problem(
    text_input: str = "",
    image_base64: Optional[str] = None,
    url_image: Optional[str] = None,
    solve_platform: str = "DeepSeek",
    solve_model: str = "deepseek-v3.2",
    thinking: bool = True,
    verify: bool = False,
    verify_platform: str = "Qwen",
    verify_model: str = "qwen3-235b-a22b",
    verify_round: int = 5,
    question_type: str = "",
    prompt: str = "",
    example: str = "",
) -> dict:
    """
    使用指定大模型对数学题进行解答，支持文本题干、图片 URL、图片 base64 三种输入。

    可选启用解析自动校验（verify=True 时启用），最多迭代 verify_round 轮。

    平台与模型必须匹配，可选组合：
      - solve_platform=DeepSeek -> solve_model=deepseek-v3.2
      - solve_platform=Qwen     -> solve_model=qwen3-235b-a22b
      - solve_platform=Gemma    -> solve_model=gemma4
    校验平台同理（Qwen / DeepSeek / Gemma / DouBao）。

    参数:
      text_input:      题干文本，纯文字题目时填这里。
      image_base64:    题目图片的 base64 字符串（不要带 data: 前缀）。
      url_image:       题目图片的可访问 URL。
      solve_platform:  解题平台，默认 DeepSeek。
      solve_model:     解题模型，需与 solve_platform 对应。
      thinking:        是否开启思维链 thinking，默认开启。
      verify:          是否在解题后做自动校验并迭代修正，默认开启。
      verify_platform: 校验平台，默认 Qwen。
      verify_model:    校验模型，默认 qwen3-235b-a22b。
      verify_round:    最大校验轮数，默认 5。
      question_type:   题型（如 "选择题"、"解答题"），可空。
      prompt:          自定义解题 prompt，可空使用服务端默认。
      example:         few-shot 示例，可空。

    返回 dict:
      ok:               请求是否成功。
      status:           解题状态。
      progress:         进度，正常完成应为 100。
      final_analysis:   最终解析（推荐展示给用户的内容）。
      analysis:         完整解析（可能含中间稿）。
      thinking_content: 思考过程（已截断）。
      verify_round:     实际校验轮数。
      cost_money:       费用消耗。
      cost_time:        耗时（秒）。
      solve_log:        中间日志（已截断）。
    """
    payload: dict[str, Any] = {
        "text_input": text_input,
        "solve_platform": solve_platform,
        "solve_model": solve_model,
        "thinking": thinking,
        "verify": verify,
        "verify_platform": verify_platform,
        "verify_model": verify_model,
        "verify_round": verify_round,
        "question_type": question_type,
    }
    if image_base64 is not None:
        payload["image_base64"] = image_base64
    if url_image is not None:
        payload["url_image"] = url_image
    if prompt:
        payload["prompt"] = prompt
    if example:
        payload["example"] = example

    try:
        async with httpx.AsyncClient(timeout=SOLVE_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE}/api/solve_problem",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "ok": False,
            "error": f"HTTP {e.response.status_code}",
            "detail": _truncate(e.response.text, 1000),
        }
    except httpx.HTTPError as e:
        return {"ok": False, "error": "request_failed", "detail": str(e)}

    return {
        "ok": True,
        "status": data.get("status"),
        "progress": data.get("progress"),
        "final_analysis": data.get("final_analysis"),
        "analysis": data.get("analysis"),
        "thinking_content": _truncate(data.get("thinking_content"), 4000),
        "verify_round": data.get("verify_round"),
        "cost_money": data.get("cost_money"),
        "cost_time": data.get("cost_time"),
        "solve_log": _truncate(data.get("solve_log"), 4000),
    }


@mcp.tool()
async def verify_analysis(
    stem_text: str,
    analysis: str,
    verify_platform: str = "Qwen",
    verify_model: str = "qwen3-235b-a22b",
) -> dict:
    """
    对已有解析文本进行逐步骤校验，返回每一步的对错与反馈。

    适用于已经拿到一份解析（来自 solve_math_problem 或人工）后，需要独立校验
    各步骤推理是否正确的场景。

    平台与模型组合：
      - verify_platform=Qwen     -> verify_model=qwen3-235b-a22b
      - verify_platform=DeepSeek -> verify_model=deepseek-v3.2
      - verify_platform=Gemma    -> verify_model=gemma4

    参数:
      stem_text:       题目题干。
      analysis:        待校验的解析文本（建议带步骤编号）。
      verify_platform: 校验平台，默认 DouBao。
      verify_model:    校验模型，默认 doubao-seed-1-6-251015。

    返回 dict:
      ok:             请求是否成功。
      has_error:      是否存在错误步骤。
      cost:           本次校验消耗费用。
      verify_results: 列表，每项为一个步骤的校验结果，字段包括
                      step_idx, verify_status (right/error/unknown),
                      verify_message, feedback_content, verify_input。
    """
    payload = {
        "stem_text": stem_text,
        "analysis": analysis,
        "verify_platform": verify_platform,
        "verify_model": verify_model,
    }

    try:
        async with httpx.AsyncClient(timeout=VERIFY_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE}/api/analysis_verify",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "ok": False,
            "error": f"HTTP {e.response.status_code}",
            "detail": _truncate(e.response.text, 1000),
        }
    except httpx.HTTPError as e:
        return {"ok": False, "error": "request_failed", "detail": str(e)}

    return {
        "ok": True,
        "has_error": data.get("has_error", False),
        "cost": data.get("cost", 0.0),
        "verify_results": data.get("verify_results", []),
    }


@mcp.tool()
async def verify_analysis_lean4(
    problem_text: str,
    solution_text: str,
) -> dict:
    """
    使用 Lean4 形式化验证对解析做机器可证明级别的校验，与 verify_analysis 互补。

    与基于 LLM 的 verify_analysis 是两条独立的验证路径，建议并行调用，
    两路结果取并集（任一路报错即视为该步骤存疑）后再进入迭代修正。

    适用场景：代数恒等式、方程推导、数值不等式等可形式化的题型。
    对几何/应用题等难以形式化的题型，Lean4 通常会在 formalization 阶段
    直接失败（stage="formalization"），此时建议以 verify_analysis 的结果为准。

    参数:
      problem_text:  数学题题干原文。
      solution_text: 题目解析、答案或证明思路文本。

    返回 dict:
      ok:              请求层面是否成功。
      status:          后端 status（success | failed | system_error）。
      success:         conclusion.success，本次形式化校验是否通过。
      has_error:       not success，便于与 verify_analysis 的语义对齐。
      stage:           结束阶段（formalization | compile | semantic | system）。
      formalization:   { has_error, error_message, lean_code, natural_language_info }。
      compile_check:   { compile_pass, error_type, error_message }。
      semantic_check:  { semantic_pass, semantic_reason, checked_lean4_code, error_message }。
      conclusion:      { success, stage, message, first_error }。
      task_id / timestamp: 后端返回的原始字段。
    """
    payload = {"problem_text": problem_text, "solution_text": solution_text}

    try:
        async with httpx.AsyncClient(timeout=LEAN4_VERIFY_TIMEOUT) as client:
            resp = await client.post(
                f"{LEAN4_API_BASE}/verify",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "ok": False,
            "error": f"HTTP {e.response.status_code}",
            "detail": _truncate(e.response.text, 1000),
        }
    except httpx.HTTPError as e:
        return {"ok": False, "error": "request_failed", "detail": str(e)}

    conclusion = data.get("conclusion") or {}
    success = bool(conclusion.get("success", False))
    return {
        "ok": True,
        "task_id": data.get("task_id"),
        "timestamp": data.get("timestamp"),
        "status": data.get("status"),
        "success": success,
        "has_error": not success,
        "stage": conclusion.get("stage"),
        "formalization": data.get("formalization") or {},
        "compile_check": data.get("compile_check") or {},
        "semantic_check": data.get("semantic_check") or {},
        "conclusion": conclusion,
    }


@mcp.tool()
async def ping() -> dict:
    """检查解题/校验后端是否可达，返回 HTTP 状态码与简短回包。"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{API_BASE}/health", headers=_headers())
        return {
            "ok": resp.status_code < 500,
            "status_code": resp.status_code,
            "text": _truncate(resp.text, 500),
            "api_base": API_BASE,
        }
    except httpx.HTTPError as e:
        return {"ok": False, "error": str(e), "api_base": API_BASE}


if __name__ == "__main__":
    mcp.run()
