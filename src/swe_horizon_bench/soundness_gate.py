"""Two-sided soundness gate and oracle-patch topology classifiers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable


def parse_patch_hunks(patch: str) -> dict[str, list[tuple[int, int]]]:
    files: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cur = None
    for line in (patch or "").splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
        elif line.startswith("@@") and cur:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2) or 1)
                files[cur].append((start, start + max(length, 1) - 1))
    return dict(files)


def classify_topology(patch: str) -> dict:
    """Return S1/S2/S3, F1/F2/F3, max_hunk_line, delta_L, any_file_delta_L_gt_150."""
    hunks = parse_patch_hunks(patch)
    # Prefer functional code files when present
    code = {
        f: hs
        for f, hs in hunks.items()
        if f.endswith(".py")
        and not f.startswith("tests/")
        and not f.startswith("docs/")
        and not f.startswith("docs_src/")
    }
    use = code or hunks
    n_files = len(use)
    n_hunks = sum(len(hs) for hs in use.values())
    max_hunk_line = 0
    max_delta = 0
    any_gt = False
    for hs in use.values():
        if not hs:
            continue
        starts = [a for a, _ in hs]
        max_hunk_line = max(max_hunk_line, max(starts))
        if len(starts) >= 2:
            d = max(starts) - min(starts)
            max_delta = max(max_delta, d)
            if d > 150:
                any_gt = True
        if max(starts) > 150:
            any_gt = any_gt or (n_files == 1 and n_hunks == 1 and max(starts) > 150)
    if n_files >= 2:
        topology = "S3_multi_file"
    elif n_hunks >= 2:
        topology = "S2_single_file_multi_hunk"
    else:
        topology = "S1_single_hunk"
    if n_hunks <= 3:
        tier = "F1"
    elif n_hunks <= 8:
        tier = "F2"
    else:
        tier = "F3"
    return {
        "topology": topology,
        "hunk_tier": tier,
        "n_gold_files": n_files,
        "n_gold_hunks": n_hunks,
        "max_hunk_line": max_hunk_line,
        "delta_L": max_delta,
        "any_file_delta_L_gt_150": bool(any_gt or max_delta > 150 or (topology.startswith("S1") and max_hunk_line > 150)),
        "files": sorted(use.keys()),
    }


def two_sided_sound(
    empty_pass: bool,
    oracle_pass: bool,
    candidate_pass: bool | None = None,
) -> dict:
    """Sound gate: empty must fail and oracle must pass (candidate optional)."""
    usable = (empty_pass is False) and (oracle_pass is True)
    out = {
        "empty_pass": bool(empty_pass),
        "oracle_pass": bool(oracle_pass),
        "usable": usable,
        "sound_rule": "empty_pass==False AND oracle_pass==True",
    }
    if candidate_pass is not None:
        out["candidate_pass"] = bool(candidate_pass)
        out["candidate_sound_solve"] = usable and bool(candidate_pass)
    return out


def run_two_sided(
    run_empty: Callable[[], bool],
    run_oracle: Callable[[], bool],
    run_candidate: Callable[[], bool] | None = None,
) -> dict:
    empty = bool(run_empty())
    oracle = bool(run_oracle())
    cand = bool(run_candidate()) if run_candidate else None
    return two_sided_sound(empty, oracle, cand)
