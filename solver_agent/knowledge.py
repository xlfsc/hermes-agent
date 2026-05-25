"""Shared knowledge base for the evolutionary solver and the live solver.

Layout (under ``solver_agent/knowledge_base/``)::

    experiences.jsonl   -- one JSON object per row, full experience record
    signatures.jsonl    -- compact rows used for fast keyword pre-filtering
                           and LLM-based ranking

The store is append-only on disk. A small in-process cache invalidates itself
when the underlying file's mtime changes, so multiple agents can read/write
concurrently without sharing state through memory.

A single experience captures: the problem, the verified-best solution (when
known), the reference answer (if supplied), error patterns observed, key
insights to reuse next time, and per-model observations. Both correct and
incorrect attempts are stored — successful runs reinforce strategies, failed
runs warn the next attempt away from known traps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient


logger = logging.getLogger(__name__)


_DEFAULT_KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


def _kb_dir() -> Path:
    raw = os.getenv("SOLVER_KB_DIR")
    return Path(raw).resolve() if raw else _DEFAULT_KB_DIR


def _experiences_path() -> Path:
    return _kb_dir() / "experiences.jsonl"


def _signatures_path() -> Path:
    return _kb_dir() / "signatures.jsonl"


def _ensure_kb_dir() -> Path:
    path = _kb_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ErrorPattern:
    model: str = ""
    pattern: str = ""
    correction: str = ""


@dataclass
class Experience:
    id: str
    ts: str
    problem: str
    problem_type: str = ""
    keywords: List[str] = field(default_factory=list)
    reference_answer: str = ""
    final_correct: bool = False
    rounds_used: int = 0
    key_insights: List[str] = field(default_factory=list)
    error_patterns: List[ErrorPattern] = field(default_factory=list)
    best_solution: str = ""
    final_answer: str = ""
    model_observations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["error_patterns"] = [asdict(p) if isinstance(p, ErrorPattern) else p for p in self.error_patterns]
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Experience":
        eps = [ErrorPattern(**p) if isinstance(p, dict) else p for p in raw.get("error_patterns", [])]
        return cls(
            id=raw.get("id", ""),
            ts=raw.get("ts", ""),
            problem=raw.get("problem", ""),
            problem_type=raw.get("problem_type", ""),
            keywords=list(raw.get("keywords", [])),
            reference_answer=raw.get("reference_answer", ""),
            final_correct=bool(raw.get("final_correct", False)),
            rounds_used=int(raw.get("rounds_used", 0)),
            key_insights=list(raw.get("key_insights", [])),
            error_patterns=eps,
            best_solution=raw.get("best_solution", ""),
            final_answer=raw.get("final_answer", ""),
            model_observations=dict(raw.get("model_observations", {})),
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"experiences": [], "mtime": 0.0, "path": ""}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_experiences(force_reload: bool = False) -> List[Experience]:
    path = _experiences_path()
    with _cache_lock:
        try:
            mtime = path.stat().st_mtime if path.exists() else 0.0
        except OSError:
            mtime = 0.0
        cached_path = _cache.get("path")
        if (
            not force_reload
            and cached_path == str(path)
            and abs(_cache.get("mtime", 0.0) - mtime) < 1e-6
            and _cache.get("experiences") is not None
        ):
            return list(_cache["experiences"])

        rows = _read_jsonl(path)
        experiences = [Experience.from_dict(r) for r in rows]
        _cache["experiences"] = experiences
        _cache["mtime"] = mtime
        _cache["path"] = str(path)
        logger.info(
            "加载经验库 | 路径=%s | 条目数=%d | 强制重载=%s",
            path, len(experiences), force_reload,
        )
        return list(experiences)


def add_experience(exp: Experience) -> Experience:
    _ensure_kb_dir()
    full_path = _experiences_path()
    sig_path = _signatures_path()
    _append_jsonl(full_path, exp.to_dict())

    sig = {
        "id": exp.id,
        "ts": exp.ts,
        "problem_type": exp.problem_type,
        "keywords": exp.keywords,
        "final_correct": exp.final_correct,
        "summary": _summarise_for_signature(exp),
    }
    _append_jsonl(sig_path, sig)

    with _cache_lock:
        _cache["mtime"] = 0.0
        _cache["path"] = ""
    logger.info(
        "经验已写入知识库 | id=%s | 题型=%s | 是否正确=%s",
        exp.id, exp.problem_type, exp.final_correct,
    )
    return exp


def _summarise_for_signature(exp: Experience, limit: int = 220) -> str:
    parts: List[str] = []
    stem = exp.problem.strip().replace("\n", " ")
    if stem:
        parts.append(f"题: {stem[:120]}")
    if exp.key_insights:
        parts.append("洞察: " + "; ".join(exp.key_insights[:2]))
    if exp.error_patterns:
        ep_strs = []
        for p in exp.error_patterns[:2]:
            if isinstance(p, ErrorPattern):
                ep_strs.append(f"{p.model}:{p.pattern}")
            elif isinstance(p, dict):
                ep_strs.append(f"{p.get('model','?')}:{p.get('pattern','')}")
        if ep_strs:
            parts.append("坑: " + "; ".join(ep_strs))
    summary = " | ".join(parts)
    return summary[:limit]


# ---------------------------------------------------------------------------
# ID & feature extraction
# ---------------------------------------------------------------------------


def make_experience_id(problem: str, ts: str) -> str:
    h = hashlib.sha1(f"{problem}|{ts}".encode("utf-8")).hexdigest()[:12]
    return f"exp_{h}"


_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "三角函数": ["sin", "cos", "tan", "三角", "正弦", "余弦", "正切", "辅助角"],
    "几何": ["三角形", "正方形", "圆", "椭圆", "抛物线", "双曲线", "向量", "面积", "周长", "体积"],
    "概率": ["概率", "随机", "期望", "分布", "抽取"],
    "数列": ["数列", "等差", "等比", "通项", "前n项和", "Sn"],
    "导数": ["导数", "切线", "单调", "极值", "最值"],
    "不等式": ["不等式", "≥", "≤", ">", "<"],
    "立体几何": ["二面角", "异面直线", "棱柱", "棱锥", "球", "正方体"],
    "解析几何": ["焦点", "准线", "渐近线", "离心率", "直线方程"],
    "函数": ["函数", "值域", "定义域", "奇偶", "周期"],
    "组合": ["排列", "组合", "C(", "A("],
}


def infer_problem_type(problem: str) -> str:
    text = problem
    for topic, kws in _TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                return topic
    return ""


def extract_keywords(problem: str, limit: int = 12) -> List[str]:
    text = problem
    found: List[str] = []
    seen: set[str] = set()
    for kws in _TOPIC_KEYWORDS.values():
        for kw in kws:
            if kw in text and kw not in seen:
                found.append(kw)
                seen.add(kw)
    for token in re.findall(r"[A-Za-z]{2,}", text):
        low = token.lower()
        if low not in seen:
            found.append(low)
            seen.add(low)
        if len(found) >= limit:
            break
    return found[:limit]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _keyword_score(query_kws: List[str], exp: Experience) -> int:
    if not query_kws:
        return 0
    score = 0
    exp_kw_set = {k.lower() for k in exp.keywords}
    for kw in query_kws:
        if kw.lower() in exp_kw_set:
            score += 2
        elif kw and kw in exp.problem:
            score += 1
    return score


def keyword_prefilter(problem: str, candidates_limit: int = 12) -> List[Experience]:
    query_kws = extract_keywords(problem)
    query_type = infer_problem_type(problem)
    experiences = load_experiences()
    if not experiences:
        logger.info("关键字预筛 | 经验库为空，跳过")
        return []
    scored: List[tuple[int, Experience]] = []
    for exp in experiences:
        score = _keyword_score(query_kws, exp)
        if query_type and exp.problem_type == query_type:
            score += 3
        if score > 0:
            scored.append((score, exp))
    scored.sort(key=lambda t: (t[0], t[1].ts), reverse=True)
    out = [exp for _, exp in scored[:candidates_limit]]
    logger.info(
        "关键字预筛完成 | 题型=%s | 关键字数=%d | 候选总数=%d | 入选=%d",
        query_type or "未识别", len(query_kws), len(experiences), len(out),
    )
    return out


def llm_rank(
    problem: str,
    candidates: List[Experience],
    *,
    top_k: int = 3,
    llm: Optional[LLMClient] = None,
) -> List[Experience]:
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    llm = llm or LLMClient()

    options_lines: List[str] = []
    for idx, exp in enumerate(candidates):
        snippet = _summarise_for_signature(exp, limit=300)
        options_lines.append(f"[{idx}] type={exp.problem_type or '-'} :: {snippet}")
    options = "\n".join(options_lines)

    system = (
        "你是数学解题经验检索助手。给定一道当前题目以及若干历史经验摘要，"
        "选出对当前题目最具参考价值的若干条经验。只考虑解题方法、常见陷阱、"
        "题型相似度，不要凭借表面字符匹配。严格输出 JSON。"
    )
    user = (
        f"当前题目:\n{problem.strip()[:1200]}\n\n"
        f"候选经验列表:\n{options}\n\n"
        f"请挑选最相关的至多 {top_k} 条，按相关度从高到低排序。"
        "返回 JSON 形如 {\"selected\": [<index>, ...]}。"
        "若没有任何条目相关，则返回 {\"selected\": []}。"
    )

    logger.info(
        "调用 LLM 重排经验 | 候选数=%d | top_k=%d", len(candidates), top_k,
    )
    try:
        data = llm.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as exc:
        logger.warning("LLM 重排失败，回退到关键字顺序 | 错误=%s", exc)
        return candidates[:top_k]

    selected_idx = data.get("selected") if isinstance(data, dict) else None
    if not isinstance(selected_idx, list):
        logger.warning("LLM 重排返回格式异常，使用关键字顺序前 %d 条", top_k)
        return candidates[:top_k]

    out: List[Experience] = []
    seen: set[int] = set()
    for raw in selected_idx:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(candidates) and i not in seen:
            out.append(candidates[i])
            seen.add(i)
        if len(out) >= top_k:
            break
    logger.info("LLM 重排完成 | 最终保留=%d", len(out) or min(len(candidates), top_k))
    return out or candidates[:top_k]


def retrieve_experiences(
    problem: str,
    *,
    top_k: int = 3,
    use_llm: bool = True,
    llm: Optional[LLMClient] = None,
) -> List[Experience]:
    """Two-stage retrieval: keyword pre-filter → LLM rerank."""

    candidates = keyword_prefilter(problem, candidates_limit=12)
    if not candidates:
        logger.info("经验检索结束 | 关键字预筛无命中")
        return []
    if not use_llm or len(candidates) <= top_k:
        out = candidates[:top_k]
        logger.info(
            "经验检索结束 | 跳过 LLM 重排 | 返回=%d (use_llm=%s)",
            len(out), use_llm,
        )
        return out
    return llm_rank(problem, candidates, top_k=top_k, llm=llm)


# ---------------------------------------------------------------------------
# Injection text builders
# ---------------------------------------------------------------------------


def build_injection_text(experiences: List[Experience]) -> Dict[str, str]:
    """Render retrieved experiences into prompt/example text fragments."""

    if not experiences:
        return {"prompt": "", "example": ""}

    insight_lines: List[str] = []
    pitfall_lines: List[str] = []
    for exp in experiences:
        for ins in exp.key_insights:
            ins = ins.strip()
            if ins:
                insight_lines.append(f"- {ins}")
        for ep in exp.error_patterns:
            model = ep.model if isinstance(ep, ErrorPattern) else ep.get("model", "")
            pattern = ep.pattern if isinstance(ep, ErrorPattern) else ep.get("pattern", "")
            correction = ep.correction if isinstance(ep, ErrorPattern) else ep.get("correction", "")
            line = f"- [{model}] {pattern}".strip()
            if correction:
                line += f" → {correction}"
            pitfall_lines.append(line)

    prompt_parts: List[str] = ["[历史经验提示]"]
    if insight_lines:
        prompt_parts.append("可参考的关键思路:")
        prompt_parts.extend(insight_lines[:8])
    if pitfall_lines:
        prompt_parts.append("已知易错点（务必规避）:")
        prompt_parts.extend(pitfall_lines[:8])
    if len(prompt_parts) == 1:
        prompt_text = ""
    else:
        prompt_text = "\n".join(prompt_parts)

    best_example = ""
    for exp in experiences:
        if exp.final_correct and exp.best_solution:
            best_example = (
                f"[相似历史正确题例]\n题目: {exp.problem.strip()[:400]}\n"
                f"解法要点: {exp.best_solution.strip()[:1200]}"
            )
            break

    return {"prompt": prompt_text, "example": best_example}


def merge_prompt(existing: str, addition: str) -> str:
    a = (existing or "").strip()
    b = (addition or "").strip()
    if not b:
        return a
    if not a:
        return b
    return f"{a}\n\n{b}"


# ---------------------------------------------------------------------------
# Convenience for callers that want a one-shot retrieval + injection bundle
# ---------------------------------------------------------------------------


def fetch_injection(
    problem: str,
    *,
    top_k: int = 3,
    use_llm: bool = True,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Retrieve relevant experiences and pre-render prompt/example text."""

    exps = retrieve_experiences(problem, top_k=top_k, use_llm=use_llm, llm=llm)
    inj = build_injection_text(exps)
    inj["matched_ids"] = [e.id for e in exps]
    inj["matched_count"] = len(exps)
    logger.info(
        "经验注入构建完成 | 命中=%d | prompt长度=%d | example长度=%d",
        len(exps), len(inj.get("prompt") or ""), len(inj.get("example") or ""),
    )
    return inj


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
