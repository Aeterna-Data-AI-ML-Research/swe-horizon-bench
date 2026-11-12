"""Unit tests for swe-horizon-bench (stdlib + pytest)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from swe_horizon_bench.ast_horizon_graph import ASTHorizonGraph, localize_windows
from swe_horizon_bench.kbench_tasks import (
    TASK_NAMES,
    _KBENCH_BACKEND,
    load_ood_dataframe,
    run_kbench_suite,
    swe_horizon_b1_vs_ast_localization,
    swe_horizon_punica_budget_feasibility,
    swe_horizon_w2_inline_repro_guard,
)
from swe_horizon_bench.ms_sft_selector import j_ms, select_ms_sft, summarize_policies
from swe_horizon_bench.soundness_gate import classify_topology, two_sided_sound


def test_two_sided_sound():
    ok = two_sided_sound(False, True, True)
    assert ok["usable"] and ok["candidate_sound_solve"]
    bad = two_sided_sound(True, True, True)
    assert not bad["usable"]


def test_classify_topology_s2_deep(tmp_path: Path):
    patch = textwrap.dedent(
        """\
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -10,1 +10,2 @@
         a
        +b
        @@ -400,1 +400,2 @@
         c
        +d
        """
    )
    c = classify_topology(patch)
    assert c["topology"].startswith("S2")
    assert c["delta_L"] >= 150
    assert c["any_file_delta_L_gt_150"]


def test_ms_sft_prefers_short_low_token():
    trajs = [
        {"task_id": "t1", "sound": True, "tokens": 8000, "steps": 40, "hunks": 5, "gold_hunks": 2, "scratch": False},
        {"task_id": "t1", "sound": True, "tokens": 2000, "steps": 12, "hunks": 1, "gold_hunks": 2, "scratch": False},
        {"task_id": "t2", "sound": True, "tokens": 3000, "steps": 20, "hunks": 2, "gold_hunks": 2, "scratch": True},
        {"task_id": "t2", "sound": True, "tokens": 3100, "steps": 18, "hunks": 2, "gold_hunks": 2, "scratch": False},
    ]
    assert j_ms(True, 2000, 12, 1, 2, False) > j_ms(True, 8000, 40, 5, 2, False)
    sel = select_ms_sft(trajs)
    assert sel["t1"]["tokens"] == 2000
    assert sel["t2"]["scratch"] is False
    summ = summarize_policies(trajs, decode_tok_s=11.01)
    assert summ["policy_ms_sft"]["mean_completion_tokens"] < summ["policy_task_uniform"]["mean_completion_tokens"]


def test_ast_horizon_graph_localize(tmp_path: Path):
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .core import DeepHelper\n")
    (pkg / "core.py").write_text(
        "\n".join(
            [
                "def shallow():",
                "    return 1",
                "",
                *["# pad"] * 200,
                "class DeepHelper:",
                "    def compute_token_overlap(self, x):",
                "        return x",
                "",
                *["# pad2"] * 50,
                "def sibling_method(y):",
                "    return y",
                "",
            ]
        )
    )
    g = ASTHorizonGraph(pkg, window_lines=150)
    g.index()
    wins = g.localize("DeepHelper compute_token_overlap fails on empty input", seed_file="core.py", k=3)
    assert wins
    assert wins[0].path.endswith("core.py")
    # deep symbol should pull a non-header window or still cover via K>=2
    lines_covered = any(w.start_line > 100 for w in wins) or any(
        w.start_line <= 220 <= w.end_line for w in wins
    )
    assert lines_covered or wins[0].end_line >= 150
    dicts = localize_windows(pkg, "DeepHelper compute_token_overlap", seed_file="core.py", k=2)
    assert len(dicts) >= 1


def test_ast_deep_s2_header_bridge_and_attr_coupling(tmp_path: Path):
    """Deep S2 (ΔL>150): K=1 header-bridge ≥ B1; K=2 covers deep self.<attr> companion."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # Module header defines shared state; deep method mutates self._token_cache (ΔL≫150)
    body = [
        "TOKEN_OVERLAP_THRESHOLD = 0.5",
        "",
        "class OverlapEngine:",
        "    def __init__(self):",
        "        self._token_cache = {}",
        "        self._overlap_mode = 'strict'",
        "",
        "    def reset_cache(self):",
        "        self._token_cache = {}",
        "",
        *["    # pad"] * 180,
        "    def compute_token_overlap(self, left, right):",
        "        # deep companion hunk: reads header-coupled attrs",
        "        if self._overlap_mode == 'strict':",
        "            self._token_cache[(left, right)] = left == right",
        "        return self._token_cache.get((left, right))",
        "",
    ]
    (pkg / "overlap.py").write_text("\n".join(body))
    g = ASTHorizonGraph(pkg, window_lines=150)
    g.index()
    assert any(g.attrs.values()), "expected self.<attr> edges"
    wins1 = g.localize(
        "OverlapEngine compute_token_overlap fails when _token_cache / _overlap_mode disagree",
        seed_file="overlap.py",
        k=1,
    )
    assert wins1 and wins1[0].start_line == 1, "K=1 should header-bridge when deep is not decisive"
    wins2 = g.localize(
        "OverlapEngine compute_token_overlap fails when _token_cache / _overlap_mode disagree",
        seed_file="overlap.py",
        k=2,
    )
    assert len(wins2) >= 2
    # one window covers deep method (~line 200+)
    deep_hit = any(w.start_line > 150 or w.end_line > 180 for w in wins2)
    assert deep_hit, "K=2 should include deep intra-file companion via attr/IDF coupling"


def test_kbench_tasks_offline_suite():
    """Native @kbench.task suite runs offline via shim on a tiny synthetic OOD frame."""
    assert _KBENCH_BACKEND in {"offline_shim", "kaggle_benchmarks"}
    assert len(TASK_NAMES) == 3
    # Minimal synthetic OOD-like rows (no holdout IDs)
    rows = [
        {
            "task_id": "ood_synth_s2_deep",
            "repo": "packaging",
            "topology": "S2_single_file_multi_hunk",
            "problem_statement": "compute_token_overlap fails for empty cache",
            "gold_patch": (
                "--- a/pkg/core.py\n+++ b/pkg/core.py\n"
                "@@ -10,1 +10,2 @@\n a\n+b\n"
                "@@ -200,1 +200,2 @@\n c\n+d\n"
            ),
            "seed_file": "pkg/core.py",
            "source_abs": {},
            "delta_L": 190,
            "any_file_delta_L_gt_150": True,
            "n_gold_hunks": 2,
            "max_hunk_line": 200,
        },
        {
            "task_id": "ood_synth_s1_shallow",
            "repo": "click",
            "topology": "S1_single_hunk",
            "problem_statement": "shallow helper returns wrong flag",
            "gold_patch": "--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -20,1 +20,2 @@\n x\n+y\n",
            "seed_file": "pkg/a.py",
            "source_abs": {},
            "delta_L": 0,
            "any_file_delta_L_gt_150": False,
            "n_gold_hunks": 1,
            "max_hunk_line": 20,
        },
    ]
    w2 = swe_horizon_w2_inline_repro_guard(df=rows)
    assert w2["passed"] and w2["inline_pass_rate_on_need"] >= 0.99
    pun = swe_horizon_punica_budget_feasibility(df=rows)
    assert pun["passed"] and pun["frac_ms_dominates_verbose"] >= 0.99
    # Localization task uses AST vs B1; may materialize synthetic files
    loc = swe_horizon_b1_vs_ast_localization(df=rows)
    assert loc["passed"] and loc["n"] == 2
    board = run_kbench_suite(df=rows)
    if hasattr(board, "shape"):
        assert board.shape[0] == 3
        assert bool(board["passed"].all())
    else:
        assert len(board) == 3
        assert all(x["passed"] for x in board)


def test_load_ood_dataframe_if_present():
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "swe_horizon_bench_ood_60.jsonl"
    if not p.is_file():
        return
    df = load_ood_dataframe(p)
    n = len(df) if hasattr(df, "__len__") else len(list(df))
    assert n == 60
