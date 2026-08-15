"""批量训练脚本：从 Excel 读取 (题干, 参考答案) 对，并行调用 evo_solver.train()。

用法:
    python -m evo_solver_agent.test.test_batch_train_xls \
        --input problems.xlsx \
        [--output-dir ./train_results] \
        [--problem-col 题干文本] \
        [--answer-col 参考答案] \
        [--sheet 0] \
        [--max-workers 2]
"""

from __future__ import annotations

import argparse
import json
import logging.config
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evo_solver_agent.evo_solver import train

with open('../logging.yaml', mode='r', encoding='utf-8') as config_file:
    logging_config = yaml.load(stream=config_file, Loader=yaml.FullLoader)
    logging.config.dictConfig(config=logging_config)

logger = logging.getLogger(__name__)


def load_problems(
        path: str, sheet: int | str, problem_col: str, answer_col: str
) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xls",):
        df = pd.read_excel(path, sheet_name=sheet, engine="xlrd")
    else:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    for col in (problem_col, answer_col):
        if col not in df.columns:
            raise KeyError(f"列 '{col}' 不存在，可用列: {list(df.columns)}")
    return df


def _train_one(
        idx: int, total: int, problem: str, reference_answer: str
) -> dict:
    logger.info("[%s/%s] 开始训练: %s", idx, total, problem[:80])
    try:
        result = train(problem, reference_answer)
        logger.info(
            "[%s/%s] 完成 (%.1fs) correct=%s rounds=%s",
            idx, total,
            result.get("elapsed_seconds", 0),
            result.get("correct"),
            result.get("rounds_used"),
        )
    except Exception as exc:
        logger.exception("[%s/%s] 训练失败", idx, total)
        result = {"ok": False, "correct": False, "error": str(exc)}
    result["index"] = idx
    result["problem"] = problem
    result["reference_answer"] = reference_answer
    return result


def _save_one(idx: int, result: dict, output_dir: str) -> None:
    tag = f"{idx:04d}"
    json_path = Path(output_dir) / "json" / f"{tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    md_path = Path(output_dir) / "md" / f"{tag}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    correct_mark = "CORRECT" if result.get("correct") else "INCORRECT"
    md_path.write_text(
        f"# 第 {idx} 题 [{correct_mark}]\n\n"
        f"## 题干\n\n{result.get('problem', '')}\n\n"
        f"## 参考答案\n\n{result.get('reference_answer', '')}\n\n"
        f"## Agent 最终答案\n\n{result.get('final_answer', 'N/A')}\n\n"
        f"## 轮数: {result.get('rounds_used', '?')}\n",
        encoding="utf-8",
    )


def run_batch(
        df: pd.DataFrame,
        problem_col: str,
        answer_col: str,
        output_dir: str,
        max_workers: int = 2,
) -> dict:
    total = len(df)
    t0 = time.monotonic()

    tasks = []
    for row_idx, row in df.iterrows():
        seq = int(row_idx) + 1
        problem = str(row[problem_col]).strip()
        answer = str(row[answer_col]).strip()
        if not problem or not answer:
            logger.warning("[%s/%s] 跳过空行", seq, total)
            continue
        tasks.append((seq, problem, answer))

    correct_count = 0
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_train_one, seq, total, problem, answer): seq
            for seq, problem, answer in tasks
        }
        for future in as_completed(futures):
            seq = futures[future]
            result = future.result()
            if result.get("correct"):
                correct_count += 1
            results.append(result)
            _save_one(seq, result, output_dir)

    elapsed = time.monotonic() - t0
    accuracy = correct_count / len(tasks) if tasks else 0.0
    logger.info(
        "全部完成: %s 题, 正确 %s, 准确率 %.2f%%, 总耗时 %.1fs",
        len(tasks), correct_count, accuracy * 100, elapsed,
    )

    summary = {
        "total": len(tasks),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
    }
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="批量监督训练 evo_solver_agent")
    parser.add_argument(
        "--input", required=False, help="Excel 文件路径",
        default=r"/home/gc/solve/hw/train/imo_random_200.xlsx"
    )
    parser.add_argument(
        "--output-dir", help="输出目录",
        default=r"/home/gc/solve/hw/train/train_results_imo_0521",
    )
    parser.add_argument(
        "--problem-col", help="题干列名", default="stem_cn",
    )
    parser.add_argument(
        "--answer-col", help="参考答案列名", default="answer",
    )
    parser.add_argument(
        "--sheet", help="Sheet 名或索引", default=0
    )
    parser.add_argument(
        "--max-workers", type=int, help="并发数", default=1,
    )
    args = parser.parse_args()

    sheet = int(args.sheet)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = load_problems(args.input, sheet, args.problem_col, args.answer_col)

    logger.info(
        "读取 %s 行 (sheet=%s, problem=%s, answer=%s)", len(df), sheet,
        args.problem_col, args.answer_col
    )

    run_batch(
        df, args.problem_col, args.answer_col, args.output_dir,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
