"""核心解题入口：编程式构造 AIAgent，预加载 multi-model-math-solving skill，
跑一次 chat() 拿最终回答。仿 hermes_cli/oneshot.py 的写法，但走独立 HERMES_HOME。"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLVER_HOME = REPO_ROOT / "solver_agent" / "hermes_home"
SOLVER_SKILLS_DIR = REPO_ROOT / "solver_agent" / "skills"

os.environ.setdefault("HERMES_HOME", str(SOLVER_HOME))
os.environ.setdefault("SOLVER_AGENT_SKILLS_DIR", str(SOLVER_SKILLS_DIR))
os.environ.setdefault("HERMES_YOLO_MODE", "1")
os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


class _AgentLogBridge:
    def __init__(self) -> None:
        self._buffer = ""

    def on_stream_delta(self, text: str | None) -> None:
        if text is None:
            self.flush()
            logger.info("agent turn boundary")
            return
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                logger.info("agent text: %s", line)

    def on_tool_gen_start(self, tool_name: str) -> None:
        self.flush()
        logger.info("agent preparing tool: %s", tool_name)

    def on_tool_start(self, tool_call_id: str, tool_name: str, args: dict) -> None:
        self.flush()
        logger.info(
            "agent tool start: %s id=%s args=%s",
            tool_name,
            tool_call_id,
            json.dumps(args, ensure_ascii=False, sort_keys=True),
        )

    def on_tool_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict,
        result: str,
    ) -> None:
        self.flush()
        preview = result.strip().replace("\n", "\\n")
        if len(preview) > 400:
            preview = preview[:400] + "..."
        logger.info(
            "agent tool complete: %s id=%s result=%s",
            tool_name,
            tool_call_id,
            preview,
        )

    def on_status(self, level: str, message: str) -> None:
        self.flush()
        logger.info("agent status[%s]: %s", level, message)

    def on_step(self, api_call_count: int, prev_tools: list[dict]) -> None:
        self.flush()
        tool_names = [tool.get("name") for tool in prev_tools if isinstance(tool, dict)]
        logger.info("agent step: api_call=%s previous_tools=%s", api_call_count, tool_names)

    def flush(self) -> None:
        line = self._buffer.strip()
        if line:
            logger.info("agent text: %s", line)
        self._buffer = ""


def _oneshot_clarify_callback(question: str, choices=None) -> str:
    logger.info("agent clarify requested: question=%s choices=%s", question, choices)
    if choices:
        return (
            f"[server mode: no interactive user. Pick the best option from "
            f"{choices} using your own judgment and continue.]"
        )
    return (
        "[server mode: no interactive user. Make the most reasonable assumption "
        "and continue.]"
    )


def solve(problem: str, *, quiet: bool = False) -> Dict[str, Any]:
    """同步执行一次解题。

    Args:
        problem: 题目文本。
        quiet: True 时把 stdout/stderr 重定向到 devnull（默认）。

    Returns:
        {"answer": str, "elapsed_seconds": float, "model": str}
    """
    if not problem or not problem.strip():
        raise ValueError("problem is empty")

    previous_disable_level = logging.root.manager.disable
    if quiet:
        logging.disable(logging.CRITICAL)

    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from agent.skill_commands import build_preloaded_skills_prompt
    from run_agent import AIAgent

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        effective_model = model_cfg
    else:
        effective_model = model_cfg.get("default") or model_cfg.get("model") or "gemma4"

    # resolve_runtime_provider("custom") returns None (runtime_provider.py:333),
    # so resolve via the entry name from custom_providers[0].
    custom_entries = get_compatible_custom_providers(cfg)
    if not custom_entries:
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
        # Fallback: build the runtime dict directly from the entry.
        runtime = {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": (primary.get("base_url") or "").rstrip("/"),
            "api_key": primary.get("api_key") or "no-key-required",
            "credential_pool": None,
        }

    enabled_toolsets = sorted(_get_platform_tools(cfg, "cli"))

    skill_names = cfg.get("default_skills") or ["multi-model-math-solving"]
    skills_prompt, loaded, missing = build_preloaded_skills_prompt(skill_names)
    if missing:
        raise RuntimeError(f"Skill(s) not found in {SOLVER_SKILLS_DIR}: {missing}")
    logger.info("Preloaded skills: %s", loaded)

    bridge = None if quiet else _AgentLogBridge()
    devnull = open(os.devnull, "w") if quiet else None

    started = time.monotonic()
    logger.info(
        "solve started: model=%s provider=%s api_mode=%s quiet=%s problem_chars=%s",
        effective_model,
        runtime.get("provider"),
        runtime.get("api_mode"),
        quiet,
        len(problem),
    )
    try:
        if quiet:
            ctx_stdout = redirect_stdout(devnull)
            ctx_stderr = redirect_stderr(devnull)
            ctx_stdout.__enter__()
            ctx_stderr.__enter__()

        try:
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
                clarify_callback=_oneshot_clarify_callback,
                stream_delta_callback=bridge.on_stream_delta if bridge else None,
                tool_gen_callback=bridge.on_tool_gen_start if bridge else None,
                tool_start_callback=bridge.on_tool_start if bridge else None,
                tool_complete_callback=bridge.on_tool_complete if bridge else None,
                status_callback=bridge.on_status if bridge else None,
                step_callback=bridge.on_step if bridge else None,
            )
            agent.suppress_status_output = True

            answer = agent.chat(problem) or ""
            if bridge:
                bridge.flush()
        finally:
            if quiet:
                ctx_stderr.__exit__(None, None, None)
                ctx_stdout.__exit__(None, None, None)
    finally:
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass
        if quiet:
            logging.disable(previous_disable_level)

    elapsed = time.monotonic() - started
    logger.info("solve finished: model=%s elapsed_seconds=%.3f answer_chars=%s", effective_model, elapsed, len(answer))
    return {
        "answer": answer,
        "elapsed_seconds": elapsed,
        "model": effective_model,
    }
