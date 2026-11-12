"""CLI: localize on any repo; evaluate OOD JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ast_horizon_graph import localize_windows
from .soundness_gate import classify_topology


def cmd_localize(args: argparse.Namespace) -> int:
    wins = localize_windows(
        args.repo,
        args.query,
        seed_file=args.seed,
        k=args.k,
        window_lines=args.window,
    )
    print(json.dumps({"repo": args.repo, "query": args.query, "windows": wins}, indent=2))
    return 0


def cmd_kbench_suite(args: argparse.Namespace) -> int:
    from .kbench_tasks import load_ood_dataframe, run_kbench_suite

    df = load_ood_dataframe(args.dataset)
    board = run_kbench_suite(df=df, out_json=args.out)
    if hasattr(board, "to_dict"):
        print(board.to_string(index=False))
    else:
        print(json.dumps(board, indent=2))
    print(f"WROTE {args.out}")
    return 0


def cmd_evaluate_ood(args: argparse.Namespace) -> int:
    path = Path(args.dataset)
    if not path.is_file():
        print(f"missing dataset: {path}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    from .ast_horizon_graph import ASTHorizonGraph

    def cover(hunks, wins):
        if not hunks:
            return 0.0
        cov = tot = 0
        for a, b in hunks:
            for ln in range(a, b + 1):
                tot += 1
                if any(w0 <= ln <= w1 for w0, w1 in wins):
                    cov += 1
        return cov / tot if tot else 0.0

    summary = {"n": len(rows), "by_topo": {}}
    for row in rows:
        topo = row.get("topology") or classify_topology(row.get("gold_patch") or "")["topology"]
        summary["by_topo"].setdefault(topo, {"n": 0})
        summary["by_topo"][topo]["n"] += 1
    out = Path(args.out) if args.out else path.with_name("swe_horizon_bench_ood_eval.json")
    # If precomputed eval exists alongside, prefer echo; else write topology counts
    pre = path.with_name("swe_horizon_bench_ood_eval.json")
    if pre.is_file() and not args.out:
        print(pre.read_text())
        return 0
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swe-horizon-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    loc = sub.add_parser("localize", help="Rank K read windows for an issue on any repo")
    loc.add_argument("--repo", required=True)
    loc.add_argument("--query", required=True)
    loc.add_argument("--seed", default=None, help="Optional seed relative file")
    loc.add_argument("-k", type=int, default=3)
    loc.add_argument("--window", type=int, default=150)
    loc.set_defaults(func=cmd_localize)

    ev = sub.add_parser("evaluate-ood", help="Summarize / load OOD eval JSONL")
    ev.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[2] / "data" / "swe_horizon_bench_ood_60.jsonl"),
    )
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=cmd_evaluate_ood)

    kb = sub.add_parser("kbench-suite", help="Run native @kbench.task suite on OOD (offline or live SDK)")
    kb.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[2] / "data" / "swe_horizon_bench_ood_60.jsonl"),
    )
    kb.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "data" / "swe_horizon_bench_kbench_leaderboard.json"),
    )
    kb.set_defaults(func=cmd_kbench_suite)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
