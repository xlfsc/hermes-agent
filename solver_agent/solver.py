"""核心解题入口：编程式构造 AIAgent，预加载 multi-model-math-solving skill，
跑一次 chat() 拿最终回答。仿 hermes_cli/oneshot.py 的写法，但走独立 HERMES_HOME。"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLVER_HOME = os.path.join(REPO_ROOT, "solver_agent", "hermes_home")
SOLVER_SKILLS_DIR = os.path.join(REPO_ROOT, "solver_agent", "skills")
TRACE_DIR = os.path.join(REPO_ROOT, "solver_agent", "traces")

os.environ.setdefault("HERMES_HOME", str(SOLVER_HOME))
os.environ.setdefault("SOLVER_AGENT_SKILLS_DIR", str(SOLVER_SKILLS_DIR))
# Keep the bundled MCP configuration portable across platforms and virtual
# environments.  The config is loaded after these variables are initialized.
os.environ.setdefault("SOLVER_AGENT_PYTHON", sys.executable)
os.environ.setdefault("SOLVER_AGENT_ROOT", str(REPO_ROOT))
os.environ.setdefault("HERMES_YOLO_MODE", "1")
os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
os.environ.setdefault("HERMES_STREAM_STALE_TIMEOUT", "1200")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _TracingBridge:
    """Collects structured trace events while also logging for operational visibility."""

    def __init__(self) -> None:
        self._buffer = ""
        self.events: list[dict] = []

    def _emit(self, event: dict) -> None:
        event.setdefault("ts", _now())
        self.events.append(event)

    def on_stream_delta(self, text: str | None) -> None:
        if text is None:
            self._flush_text()
            logger.info("Agent 轮次边界")
            self._emit({"type": "turn_boundary"})
            return
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                logger.info("Agent 文本输出 | 内容=%s", line)

    def on_tool_gen_start(self, tool_name: str) -> None:
        self._flush_text()
        logger.info("Agent 准备调用工具 | 工具=%s", tool_name)
        self._emit({"type": "tool_gen", "tool_name": tool_name})

    def on_tool_start(
            self, tool_call_id: str, tool_name: str, args: dict
    ) -> None:
        self._flush_text()
        logger.info(
            "Agent 工具调用开始 | 工具=%s | 调用ID=%s | 参数=%s",
            tool_name, tool_call_id,
            json.dumps(args, ensure_ascii=False, sort_keys=True),
        )
        self._emit({
            "type": "tool_start",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "args": args,
        })

    def on_tool_complete(
            self, tool_call_id: str, tool_name: str, args: dict, result: str,
    ) -> None:
        self._flush_text()
        preview = result.strip().replace("\n", "\\n")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        logger.info(
            "Agent 工具调用完成 | 工具=%s | 调用ID=%s | 返回长度=%d | 返回摘要=%s",
            tool_name,
            tool_call_id,
            len(result),
            preview[:400]
        )
        self._emit({
            "type": "tool_complete",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result_chars": len(result),
            "result_preview": preview,
        })

    def on_status(self, level: str, message: str) -> None:
        self._flush_text()
        logger.info("Agent 状态事件 | 级别=%s | 消息=%s", level, message)
        self._emit({"type": "status", "level": level, "message": message})

    def on_step(self, api_call_count: int, prev_tools: list[dict]) -> None:
        self._flush_text()
        tool_names = [t.get("name") for t in prev_tools if isinstance(t, dict)]
        logger.info(
            "Agent 推进一步 | API调用次数=%s | 上一步工具=%s",
            api_call_count,
            tool_names
        )
        self._emit(
            {
                "type": "step",
                "api_call": api_call_count,
                "prev_tools": prev_tools
            }
        )

    def _flush_text(self) -> None:
        text = self._buffer.strip()
        if text:
            logger.info("Agent 文本输出 | 内容=%s", text)
            self._emit({"type": "text", "content": text})
        self._buffer = ""

    def flush(self) -> None:
        self._flush_text()


def _make_clarify_callback(bridge: _TracingBridge | None):
    def _callback(question: str, choices=None) -> str:
        logger.info(
            "Agent 请求澄清 | 问题=%s | 备选=%s",
            question,
            choices
        )
        if bridge:
            bridge._emit(
                {"type": "clarify", "question": question, "choices": choices}
            )
        if choices:
            return (
                f"[server mode: no interactive user. Pick the best option from "
                f"{choices} using your own judgment and continue.]"
            )
        return (
            "[server mode: no interactive user. Make the most reasonable assumption "
            "and continue.]"
        )

    return _callback


def solve(problem: str, *, quiet: bool = False) -> Dict[str, Any]:
    """同步执行一次解题，返回答案和 trace 信息。

    Returns:
        {"answer": str, "elapsed_seconds": float, "model": str, "trace_id": str}
    """
    if not problem or not problem.strip():
        raise ValueError("problem is empty")

    trace_id = f"trc_{uuid.uuid4().hex[:12]}"
    logger.info(
        "收到解题请求 | trace_id=%s | quiet=%s | 题干长度=%d",
        trace_id, quiet, len(problem)
    )

    previous_disable_level = logging.root.manager.disable
    if quiet:
        logging.disable(logging.CRITICAL)

    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from agent.skill_commands import build_preloaded_skills_prompt
    from run_agent import AIAgent

    cfg = load_config()
    logger.info("已加载 Hermes 配置 | HERMES_HOME=%s", os.environ.get("HERMES_HOME"))
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        effective_model = model_cfg
    else:
        effective_model = model_cfg.get("default") or model_cfg.get(
            "model") or "gemma4"
    logger.info("使用模型 | model=%s", effective_model)

    custom_entries = get_compatible_custom_providers(cfg)
    if not custom_entries:
        logger.error(
            "config.yaml 缺少 custom_providers 条目 | 期望路径=%s",
            os.path.join(SOLVER_HOME, "config.yaml"),
        )
        raise RuntimeError(
            "No custom_providers entry in config.yaml. "
            f"Expected one in {SOLVER_HOME / 'config.yaml'}."
        )
    primary = custom_entries[0]
    runtime = resolve_runtime_provider(
        requested=primary.get("name") or "custom",
        target_model=effective_model,
    )
    if not runtime.get("base_url"):
        logger.warning(
            "runtime_provider 未解析到 base_url，回退到 custom_providers 主条目"
        )
        runtime = {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": (primary.get("base_url") or "").rstrip("/"),
            "api_key": primary.get("api_key") or "no-key-required",
            "credential_pool": None,
        }
    logger.info(
        "运行时已解析 | provider=%s | api_mode=%s | base_url=%s",
        runtime.get("provider"),
        runtime.get("api_mode"),
        runtime.get("base_url"),
    )

    enabled_toolsets = sorted(_get_platform_tools(cfg, "cli"))
    logger.info("启用的工具集 | 数量=%d | 列表=%s", len(enabled_toolsets), enabled_toolsets)

    skill_names = cfg.get("default_skills") or ["multi-model-math-solving"]
    skills_prompt, loaded, missing = build_preloaded_skills_prompt(skill_names)
    if missing:
        logger.error(
            "缺失技能 | 目录=%s | 缺失=%s", SOLVER_SKILLS_DIR, missing
        )
        raise RuntimeError(
            f"Skill(s) not found in {SOLVER_SKILLS_DIR}: {missing}"
        )
    logger.info("已预加载技能 | 列表=%s", loaded)

    bridge = _TracingBridge()
    devnull = open(os.devnull, "w") if quiet else None

    started = time.monotonic()
    logger.info(
        "解题流程开始 | trace_id=%s | model=%s | provider=%s | quiet=%s | 题干长度=%s",
        trace_id, effective_model,
        runtime.get("provider"),
        quiet,
        len(problem),
    )

    conv_result: Dict[str, Any] = {}
    try:
        if quiet:
            ctx_stdout = redirect_stdout(devnull)
            ctx_stderr = redirect_stderr(devnull)
            ctx_stdout.__enter__()
            ctx_stderr.__enter__()

        try:
            logger.info("开始构建 AIAgent | trace_id=%s", trace_id)
            agent = AIAgent(
                api_key=runtime.get("api_key"),
                base_url=runtime.get("base_url"),
                provider=runtime.get("provider"),
                api_mode=runtime.get("api_mode"),
                model=effective_model,
                enabled_toolsets=enabled_toolsets,
                quiet_mode=True,
                platform="cli",
                credential_pool=runtime.get("credential_pool"),
                ephemeral_system_prompt=skills_prompt or None,
                clarify_callback=_make_clarify_callback(bridge),
                stream_delta_callback=bridge.on_stream_delta,
                tool_gen_callback=bridge.on_tool_gen_start,
                tool_start_callback=bridge.on_tool_start,
                tool_complete_callback=bridge.on_tool_complete,
                status_callback=bridge.on_status,
                step_callback=bridge.on_step,
            )
            agent.suppress_status_output = True
            try:
                _tool_names = sorted(getattr(agent, "valid_tool_names", []) or [])
                logger.info(
                    "AIAgent 工具集就绪 | trace_id=%s | 工具数=%d | 工具列表=%s",
                    trace_id, len(_tool_names), _tool_names,
                )
            except Exception:
                logger.exception("打印工具列表失败 | trace_id=%s", trace_id)
            logger.info("AIAgent 构建完成，进入对话 | trace_id=%s", trace_id)

            conv_result = agent.run_conversation(problem) or {}
            bridge.flush()
            logger.info(
                "对话结束 | trace_id=%s | completed=%s | api_calls=%s | input_tokens=%s | output_tokens=%s",
                trace_id,
                conv_result.get("completed"),
                conv_result.get("api_calls"),
                conv_result.get("input_tokens"),
                conv_result.get("output_tokens"),
            )
        finally:
            if quiet:
                ctx_stderr.__exit__(None, None, None)
                ctx_stdout.__exit__(None, None, None)
    except Exception:
        logger.exception("解题流程异常 | trace_id=%s", trace_id)
        raise
    finally:
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass
        if quiet:
            logging.disable(previous_disable_level)

    answer = conv_result.get("final_response") or ""
    elapsed = time.monotonic() - started
    logger.info(
        "解题流程结束 | trace_id=%s | 耗时=%.3fs | 答案长度=%s",
        trace_id, elapsed, len(answer)
    )

    _write_trace(
        trace_id, problem, quiet, effective_model, runtime,
        conv_result, bridge, elapsed
    )

    return {
        "answer": answer,
        "elapsed_seconds": elapsed,
        "model": effective_model,
        "trace_id": trace_id,
    }


def _write_trace(
        trace_id: str,
        problem: str,
        quiet: bool,
        model: str,
        runtime: dict,
        conv_result: dict,
        bridge: _TracingBridge,
        elapsed: float,
) -> None:
    try:
        Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
        trace = {
            "trace_id": trace_id,
            "created_at": _now(),
            "request": {
                "problem": problem,
                "quiet": quiet,
                "model": model,
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
            },
            "result": {
                "answer": conv_result.get("final_response") or "",
                "elapsed_seconds": round(elapsed, 3),
                "completed": conv_result.get("completed", False),
                "api_calls": conv_result.get("api_calls", 0),
                "input_tokens": conv_result.get("input_tokens", 0),
                "output_tokens": conv_result.get("output_tokens", 0),
                "estimated_cost_usd": conv_result.get("estimated_cost_usd",
                                                      0.0),
            },
            "events": bridge.events,
            "messages": conv_result.get("messages", []),
        }
        trace_path = Path(os.path.join(TRACE_DIR, f"{trace_id}.json"))
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("trace 已写入 | 路径=%s", trace_path)
    except Exception:
        logger.exception("trace 写入失败 | trace_id=%s", trace_id)
