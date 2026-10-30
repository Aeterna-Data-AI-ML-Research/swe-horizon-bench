"""Punica-aware Minimal-Sufficient Trajectory Selector (MS-SFT)."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable


def j_ms(
    sound: bool,
    tokens: float,
    steps: float,
    hunks: float,
    gold_hunks: float,
    scratch: bool,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    gamma: float = 0.5,
    delta: float = 2.0,
    t_max: float = 8192.0,
    n_max: float = 80.0,
) -> float:
    if not sound:
        return 0.0
    gh = max(float(gold_hunks), 1.0)
    tok = float(tokens) if tokens and tokens > 0 else float(steps) * 120.0
    return math.exp(
        -alpha * (tok / t_max)
        - beta * (float(steps) / n_max)
        - gamma * (float(hunks) / gh)
        - delta * (1.0 if scratch else 0.0)
    )


def select_ms_sft(trajectories: Iterable[dict[str, Any]]) -> dict[str, dict]:
    """Per-task argmax J_MS. Each traj needs: task_id, sound, tokens, steps, hunks, gold_hunks, scratch."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        tid = t["task_id"]
        score = j_ms(
            bool(t.get("sound", True)),
            float(t.get("tokens") or t.get("completion_tokens") or 0),
            float(t.get("steps") or 0),
            float(t.get("hunks") or 1),
            float(t.get("gold_hunks") or 1),
            bool(t.get("scratch", False)),
        )
        row = dict(t)
        row["j_ms"] = score
        by_task[tid].append(row)
    return {tid: max(rs, key=lambda r: r["j_ms"]) for tid, rs in by_task.items()}


def project_wall_hours(
    mean_tokens: float,
    *,
    decode_tok_s: float,
    n_tasks: int = 120,
    think_factor: float = 1.0,
) -> float:
    if not mean_tokens or decode_tok_s <= 0:
        return float("nan")
    return (mean_tokens / decode_tok_s) * n_tasks / 3600.0 * think_factor


def summarize_policies(
    trajectories: list[dict[str, Any]],
    *,
    decode_tok_s: float = 11.01,
    think_2048_factor: float = 0.72,
) -> dict[str, Any]:
    """Task-balanced macro summaries for uniform / gold / step-min / MS-SFT."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        if not t.get("sound", True):
            continue
        by_task[t["task_id"]].append(t)

    def macro(rows: list[dict], label: str) -> dict:
        mean_steps = statistics.mean(float(r.get("steps") or 0) for r in rows)
        mean_hunks = statistics.mean(float(r.get("hunks") or 0) for r in rows)
        mean_toks = statistics.mean(
            float(r.get("tokens") or r.get("completion_tokens") or 0) for r in rows
        )
        return {
            "label": label,
            "n_tasks": len(rows),
            "mean_steps": mean_steps,
            "mean_hunks": mean_hunks,
            "mean_completion_tokens": mean_toks,
            "proj_h120_decode_think4096": project_wall_hours(mean_toks, decode_tok_s=decode_tok_s),
            "proj_h120_decode_think2048": project_wall_hours(
                mean_toks, decode_tok_s=decode_tok_s, think_factor=think_2048_factor
            ),
        }

    uniform = []
    for tid, rs in by_task.items():
        uniform.append(
            {
                "task_id": tid,
                "steps": statistics.mean(float(r.get("steps") or 0) for r in rs),
                "hunks": statistics.mean(float(r.get("hunks") or 0) for r in rs),
                "tokens": statistics.mean(
                    float(r.get("tokens") or r.get("completion_tokens") or 0) for r in rs
                ),
            }
        )
    gold = [
        max(
            rs,
            key=lambda r: (
                1.0
                - abs(float(r.get("hunks") or 0) - float(r.get("gold_hunks") or 1))
                / max(float(r.get("gold_hunks") or 1), 1.0),
                float(r.get("tokens") or r.get("completion_tokens") or 0),
            ),
        )
        for rs in by_task.values()
    ]
    step_min = [
        min(rs, key=lambda r: (float(r.get("steps") or 0), float(r.get("tokens") or 0)))
        for rs in by_task.values()
    ]
    ms = list(select_ms_sft([dict(r, sound=True) for rs in by_task.values() for r in rs]).values())
    return {
        "policy_task_uniform": macro(uniform, "Task-Uniform"),
        "policy_full_gold": macro(gold, "Full-Gold"),
        "policy_step_minimal": macro(step_min, "Step-Minimal"),
        "policy_ms_sft": macro(ms, "MS-SFT"),
        "n_tasks": len(by_task),
        "n_traj": sum(len(v) for v in by_task.values()),
    }
