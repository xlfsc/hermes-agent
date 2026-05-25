"""单例解题测试：输入一段题干文本，调用 solver.solve() 并打印结果。

用法:
    python -m solver_agent.test.test_single_solve --problem "已知 sin x = 1/2，求 x 的最小正解"
    python -m solver_agent.test.test_single_solve --file problem.txt
    python -m solver_agent.test.test_single_solve            # 使用内置示例题
"""

from __future__ import annotations

import argparse
import json
import logging.config
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver_agent.solver import solve

with open('../logging.yaml', mode='r', encoding='utf-8') as config_file:
    logging_config = yaml.load(stream=config_file, Loader=yaml.FullLoader)
    logging.config.dictConfig(config=logging_config)

logger = logging.getLogger(__name__)

DEFAULT_PROBLEM = (
    "已知函数 f(x) = 2*sin(x) + cos(x)，求 f(x) 的最大值，并写出取得最大值时 x 的取值集合。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单例解题测试")
    parser.add_argument(
        "--problem", "-p", type=str, default=None,
        help="题干文本（与 --file 二选一，都不传则使用内置示例题）",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="题干文本文件路径（UTF-8）",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="抑制 Agent stdout/stderr",
    )
    return parser.parse_args()


def _load_problem(args: argparse.Namespace) -> str:
    if args.problem:
        return args.problem.strip()
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").strip()
    return DEFAULT_PROBLEM


def main() -> int:
    args = parse_args()
    problem = _load_problem(args)
    if not problem:
        logger.error("题干为空，无法测试")
        return 1

    logger.info("单例测试开始 | 题干长度=%d | quiet=%s", len(problem), args.quiet)
    logger.info("题干内容:\n%s", problem)

    result = solve(problem, quiet=args.quiet)

    logger.info(
        "单例测试完成 | trace_id=%s | model=%s | 耗时=%.3fs | 答案长度=%d",
        result.get("trace_id"),
        result.get("model"),
        result.get("elapsed_seconds", 0.0),
        len(result.get("answer") or ""),
    )
    print("\n========== 解题结果 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print("==============================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
