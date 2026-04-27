"""批量解题脚本：从 Excel 读取题目，逐行调用 solver.solve()。
- 题干 + 解析 拼接写入 Markdown
- 每题的完整 solve() 返回值保留在 JSON

用法:
    python -m solver_agent.test.test_batch_solve_xls \
        --input problems.xlsx \
        [--output-dir ./batch_results] \
        [--problem-col 题目] \
        [--sheet 0] \
        [--quiet]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from solver_agent.solver import solve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_problems(
        path: str, sheet: int | str, problem_col: str
) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xls",):
        df = pd.read_excel(path, sheet_name=sheet, engine="xlrd")
    else:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if problem_col not in df.columns:
        raise KeyError(
            f"列 '{problem_col}' 不存在，可用列: {list(df.columns)}"
        )
    return df


def _save_one(seq: int, problem: str, result: dict, output_dir: Path) -> None:
    tag = f"{seq: 04d}"

    md_path = output_dir / f"md/{tag}.md"
    md_path.write_text(
        f"# 第 {seq} 题\n\n"
        f"## 题干\n\n{problem}\n\n"
        f"## 解析\n\n{result.get('answer', 'N/A')}\n",
        encoding="utf-8",
    )

    json_path = output_dir / f"json/{tag}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("已写入: %s, %s", md_path.name, json_path.name)


def run_batch(
        df: pd.DataFrame, problem_col: str, quiet: bool, output_dir: Path,
) -> None:
    total = len(df)
    t0 = time.monotonic()
    seq = 0

    for idx, row in df.iterrows():
        problem = str(row[problem_col]).strip()
        if not problem:
            logger.warning("[%s/%s] 跳过空题目 (row %s)", idx + 1, total, idx)
            continue

        seq += 1
        logger.info("[%s/%s] 开始解题: %s", seq, total, problem[:80])
        try:
            result = solve(problem, quiet=quiet)
            logger.info(
                "[%s/%s] 完成 (%.1fs): %s",
                seq, total, result["elapsed_seconds"],
                str(result["answer"])[:120],
            )
        except Exception as exc:
            logger.exception("[%s/%s] 解题失败 (row %s)", seq, total, idx)
            result = {"answer": "ERROR", "error": str(exc)}

        result["index"] = int(idx)
        result["problem"] = problem
        _save_one(seq, problem, result, output_dir)

    elapsed = time.monotonic() - t0
    logger.info("全部完成: %s 题, 总耗时 %.1fs", seq, elapsed)


def main() -> None:
    sheet = 0
    input_path = r"F:\lab\hw\第一阶段\函数\hw_test_0206.xlsx"
    output_dir = Path(r"F:\lab\hw\第一阶段\函数\hermes_0428")
    problem_col = '题干文本'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_problems(input_path, sheet, problem_col)

    logger.info(
        "读取 %s 行题目 (sheet=%s, col=%s)",
        len(df),
        sheet,
        problem_col
    )

    run_batch(df, problem_col, False, output_dir)


if __name__ == "__main__":
    main()
