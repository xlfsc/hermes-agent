"""启动入口：`python solver_agent/run_server.py`

环境变量：
    SOLVER_HOST  默认 0.0.0.0
    SOLVER_PORT  默认 8765
    HERMES_HOME  自动指向 solver_agent/hermes_home/（在 solver.py 中 setdefault）
"""

from __future__ import annotations

import logging.config
import os
import sys
from pathlib import Path

import yaml

with open('logging.yaml', mode='r', encoding='utf-8') as config_file:
    logging_config = yaml.load(stream=config_file, Loader=yaml.FullLoader)
    logging.config.dictConfig(config=logging_config)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 触发 solver.py 顶部的 os.environ 设置（HERMES_HOME 等），必须在 import api 之前
import solver_agent.solver  # noqa: F401,E402

import uvicorn  # noqa: E402


def main() -> None:
    host = os.getenv("SOLVER_HOST", "0.0.0.0")
    port = int(os.getenv("SOLVER_PORT", "8765"))
    logger.info(
        f"启动 Solver 服务 | host={host} | port={port} | "
        f"HERMES_HOME={os.environ.get('HERMES_HOME')} | "
        f"stream_stale_timeout={os.environ.get('HERMES_STREAM_STALE_TIMEOUT')}s"
    )
    uvicorn.run(
        "solver_agent.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
