"""Unit tests for swe-horizon-bench (stdlib + pytest)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from swe_horizon_bench.ast_horizon_graph import ASTHorizonGraph, localize_windows
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
