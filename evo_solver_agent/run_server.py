"""启动入口：`python evo_solver_agent/run_server.py`

环境变量：
    EVO_HOST          默认 0.0.0.0
    EVO_PORT          默认 8766
    EVO_MAX_ROUNDS    默认 3
    SOLVER_API_BASE   解题/校验后端地址 (复用 solver_agent)
    SOLVER_KB_DIR     知识库目录 (默认 solver_agent/knowledge_base)
    KB_LLM_BASE_URL   反思/检索用 LLM 的 base url
    KB_LLM_MODEL      反思/检索用模型 (默认 gemma4)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import uvicorn  # noqa: E402


def main() -> None:
    host = os.getenv("EVO_HOST", "0.0.0.0")
    port = int(os.getenv("EVO_PORT", "8766"))
    uvicorn.run(
        "evo_solver_agent.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
