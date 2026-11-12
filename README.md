# SWE-HorizonBench

**AST-HorizonGraph** localization, **MS-SFT** trajectory selection, and a **60-task out-of-distribution** multi-hunk SWE benchmark — released as a standalone, pip-installable toolkit for coding-agent research on *any* Python repository.

## Install

```bash
pip install -e .
# optional
pip install -e ".[dev]"
```

Zero heavy runtime dependencies (stdlib `ast`, `json`, `pathlib`, `re`, `math`).

## Quickstart — localize on any repo

```bash
python3 -m swe_horizon_bench.cli localize \
  --repo /path/to/your/python/project \
  --query "AsyncClient timeout ignored when follow_redirects=False" \
  --seed src/client.py \
  -k 3
```

Returns up to `K` coupling-ranked **150-line** read windows from a 1-hop import/call/sibling AST graph.

## Python API

```python
from swe_horizon_bench import localize_windows, j_ms, classify_topology, two_sided_sound

wins = localize_windows("~/code/myrepo", "pluggy hook wrapper order", k=3)
topo = classify_topology(open("gold.patch").read())
gate = two_sided_sound(empty_pass=False, oracle_pass=True, candidate_pass=True)
```

## OOD benchmark

- `data/swe_horizon_bench_ood_60.jsonl` — 60 tasks across 8 external libraries  
  (`starlette`, `click`, `typer`, `urllib3`, `packaging`, `pluggy`, `anyio`, `httpcore`)  
  with **0% overlap** vs competition `{fastapi, rich, requests, httpx}` task IDs.
- Stratified (diff-classified): 18 S1 (8 ≤150 / 10 >150), 22 S2 (11 ΔL≤150 / 11 ΔL>150), 20 S3.
- `data/swe_horizon_bench_ood_eval.json` — AST-HorizonGraph Recall@1/2/3 vs B1 and sequential pagination.

```bash
python3 -m swe_horizon_bench.cli evaluate-ood \
  --dataset data/swe_horizon_bench_ood_60.jsonl
```

## Tests

```bash
pytest tests/
```

## License

Apache License 2.0 — see `LICENSE`.
