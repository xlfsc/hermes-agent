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

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

# stdio 子进程里默认没有 logging 配置，INFO 不会输出。
# 同时把日志写到 stderr（父进程会重定向到 hermes_home/logs/mcp-stderr.log）
# 和 solver_agent/log/solver_mcp_server.log（独立文件，方便直接 tail）。
_LOG_LEVEL = os.environ.get("SOLVER_MCP_LOG_LEVEL", "INFO").upper()
_LOG_DIR = Path(__file__).resolve().parent / "log"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "solver_mcp_server.log"

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s pid=%(process)d : %(message)s"
)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(_log_formatter)
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_log_formatter)

_root_logger = logging.getLogger()
_root_logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
# force=True 等价：清空已有 handler 再装
for _h in list(_root_logger.handlers):
    _root_logger.removeHandler(_h)
_root_logger.addHandler(_stderr_handler)
_root_logger.addHandler(_file_handler)

logger = logging.getLogger(__name__)
logger.info("solver_mcp_server 日志初始化完成 | 日志文件=%s | 级别=%s", _LOG_FILE, _LOG_LEVEL)


# Allow ``from solver_agent.knowledge import ...`` when this module is invoked as
# a standalone script (the MCP server is launched via ``python <path>.py``).
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

try:  # pragma: no cover - defensive: knowledge base is optional
    from solver_agent.knowledge import fetch_injection, merge_prompt  # type: ignore
    _KB_AVAILABLE = True
except Exception as _kb_exc:  # pragma: no cover
    _KB_AVAILABLE = False
    _KB_IMPORT_ERROR = _kb_exc
    logger.warning("知识库模块加载失败，将使用空注入 | 错误=%s", _kb_exc)

    def fetch_injection(*_args, **_kwargs):  # type: ignore
        return {"prompt": "", "example": "", "matched_ids": [], "matched_count": 0}

    def merge_prompt(a: str, b: str) -> str:  # type: ignore
        a = (a or "").strip()
        b = (b or "").strip()
        if not a:
            return b
        if not b:
            return a
        return f"{a}\n\n{b}"


API_BASE = os.environ.get("SOLVER_API_BASE", "http://172.168.80.46:8000").rstrip("/")
API_KEY = os.environ.get("SOLVER_API_KEY")

SOLVE_TIMEOUT = float(os.environ.get("SOLVER_SOLVE_TIMEOUT", "600"))
VERIFY_TIMEOUT = float(os.environ.get("SOLVER_VERIFY_TIMEOUT", "300"))

AUTO_EXPERIENCE = os.environ.get("SOLVER_AUTO_EXPERIENCE", "1").lower() not in {"0", "false", "no", ""}
KB_TOP_K = int(os.environ.get("SOLVER_KB_TOP_K", "3"))
KB_USE_LLM_RANK = os.environ.get("SOLVER_KB_LLM_RANK", "1").lower() not in {"0", "false", "no", ""}

PLATFORM_MODELS = {
    "Qwen":     ["qwen3-235b-a22b"],
    "DeepSeek": ["deepseek-v3.2"],
    "Gemma":    ["gemma4"],
}

mcp = FastMCP("solver")

logger.info(
    "Solver MCP server 启动 | API_BASE=%s | AUTO_EXPERIENCE=%s | KB_TOP_K=%d | KB_USE_LLM_RANK=%s | 文件=%s",
    API_BASE, AUTO_EXPERIENCE, KB_TOP_K, KB_USE_LLM_RANK, __file__,
)

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
    # --- Knowledge base auto-injection ---
    logger.info(
        "MCP 工具 solve_math_problem 调用 | 平台=%s | 模型=%s | 题干长度=%d | 校验=%s | 思维链=%s",
        solve_platform, solve_model, len(text_input or ""), verify, thinking,
    )
    effective_prompt = prompt or ""
    effective_example = example or ""
    kb_meta: dict[str, Any] = {}
    if AUTO_EXPERIENCE and text_input:
        try:
            inj = fetch_injection(text_input, top_k=KB_TOP_K, use_llm=KB_USE_LLM_RANK)
            effective_prompt = merge_prompt(effective_prompt, inj.get("prompt", ""))
            effective_example = merge_prompt(effective_example, inj.get("example", ""))
            kb_meta = {"kb_matched_ids": inj.get("matched_ids", []), "kb_matched_count": inj.get("matched_count", 0)}
            logger.info(
                "经验自动注入完成 | 命中=%d | ids=%s",
                kb_meta["kb_matched_count"], kb_meta["kb_matched_ids"],
            )
        except Exception as exc:
            logger.warning("经验自动注入失败，已忽略 | 错误=%s", exc)
            pass  # knowledge base failure must not block solving

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
    if effective_prompt:
        payload["prompt"] = effective_prompt
    if effective_example:
        payload["example"] = effective_example

    try:
        async with httpx.AsyncClient(timeout=SOLVE_TIMEOUT) as client:
            logger.info(
                "向后端发起解题请求 | URL=%s/api/solve_problem | 超时=%.0fs, 输入: %s",
                API_BASE, SOLVE_TIMEOUT,
                json.dumps(payload, ensure_ascii=False)
            )
            resp = await client.post(
                f"{API_BASE}/api/solve_problem",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "解题后端返回 HTTP 错误 | 状态码=%d | 详情=%s",
            e.response.status_code, _truncate(e.response.text, 300),
        )
        return {
            "ok": False,
            "error": f"HTTP {e.response.status_code}",
            "detail": _truncate(e.response.text, 1000),
        }
    except httpx.HTTPError as e:
        logger.warning("解题后端请求失败 | 错误=%s", e)
        return {"ok": False, "error": "request_failed", "detail": str(e)}

    logger.info(
        "解题后端返回 | status=%s | progress=%s | verify_round=%s | cost_time=%s",
        data.get("status"), data.get("progress"),
        data.get("verify_round"), data.get("cost_time"),
    )
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
        **kb_meta,
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

    logger.info(
        "MCP 工具 verify_analysis 调用 | 平台=%s | 模型=%s | 题干长度=%d | 解析长度=%d",
        verify_platform, verify_model, len(stem_text or ""), len(analysis or ""),
    )
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
        logger.warning(
            "校验后端返回 HTTP 错误 | 状态码=%d | 详情=%s",
            e.response.status_code, _truncate(e.response.text, 300),
        )
        return {
            "ok": False,
            "error": f"HTTP {e.response.status_code}",
            "detail": _truncate(e.response.text, 1000),
        }
    except httpx.HTTPError as e:
        logger.warning("校验后端请求失败 | 错误=%s", e)
        return {"ok": False, "error": "request_failed", "detail": str(e)}

    logger.info(
        "校验后端返回 | has_error=%s | 步骤数=%d | 费用=%s",
        data.get("has_error", False),
        len(data.get("verify_results") or []),
        data.get("cost", 0.0),
    )
    return {
        "ok": True,
        "has_error": data.get("has_error", False),
        "cost": data.get("cost", 0.0),
        "verify_results": data.get("verify_results", []),
    }


@mcp.tool()
async def ping() -> dict:
    """检查解题/校验后端是否可达，返回 HTTP 状态码与简短回包。"""
    logger.info("MCP 工具 ping 调用 | URL=%s/health", API_BASE)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{API_BASE}/health", headers=_headers())
        logger.info("ping 完成 | 状态码=%d", resp.status_code)
        return {
            "ok": resp.status_code < 500,
            "status_code": resp.status_code,
            "text": _truncate(resp.text, 500),
            "api_base": API_BASE,
        }
    except httpx.HTTPError as e:
        logger.warning("ping 失败 | 错误=%s", e)
        return {"ok": False, "error": str(e), "api_base": API_BASE}


@mcp.tool()
async def retrieve_experiences(problem: str, top_k: int = 3) -> dict:
    """
    从共享知识库检索与给定题目相关的历史经验。

    返回与该题最相关的若干条经验摘要，可用于人工审阅或拼装到自定义 prompt 中。
    注意：``solve_math_problem`` 在 SOLVER_AUTO_EXPERIENCE=1 时已经会自动检索并
    注入经验，本工具用于 Agent 想显式查看注入了哪些经验、或基于经验进一步调整
    策略时使用。

    参数:
      problem: 题干文本。
      top_k:   返回条数上限。

    返回 dict:
      ok:             是否成功。
      matched_count:  实际匹配的条数。
      matched_ids:    经验 id 列表。
      prompt:         可直接拼接到 solve_math_problem 的 prompt 文本。
      example:        可直接拼接到 solve_math_problem 的 example 文本。
    """
    if not _KB_AVAILABLE:
        logger.warning("MCP 工具 retrieve_experiences 调用但知识库不可用")
        return {"ok": False, "error": "knowledge_base_unavailable"}
    logger.info(
        "MCP 工具 retrieve_experiences 调用 | 题干长度=%d | top_k=%d",
        len(problem or ""), top_k,
    )
    try:
        inj = fetch_injection(problem, top_k=top_k, use_llm=KB_USE_LLM_RANK)
    except Exception as exc:
        logger.warning("retrieve_experiences 异常 | 错误=%s", exc)
        return {"ok": False, "error": str(exc)}
    logger.info(
        "retrieve_experiences 完成 | 命中=%d | ids=%s",
        inj.get("matched_count", 0), inj.get("matched_ids", []),
    )
    return {
        "ok": True,
        "matched_count": inj.get("matched_count", 0),
        "matched_ids": inj.get("matched_ids", []),
        "prompt": inj.get("prompt", ""),
        "example": inj.get("example", ""),
    }


if __name__ == "__main__":
    mcp.run()
