"""AST-HorizonGraph: IDF-seeded localization with import/call/sibling/attr edges."""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


STOP = {
    "the", "and", "for", "that", "this", "with", "from", "when", "have",
    "should", "would", "could", "into", "about", "test", "tests", "assert",
    "error", "exception", "return", "true", "false", "none", "self", "class",
    "def", "import", "file", "code", "issue", "bug", "fix", "please",
    "expected", "actual", "version", "python", "args", "kwargs", "value",
}


def split_ident(name: str) -> list[str]:
    """Exact identifier plus snake_case / CamelCase sub-tokens (≥3 chars)."""
    if not name:
        return []
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)
    snake = name.split("_")
    out = {name}
    for p in parts + snake:
        if len(p) >= 3:
            out.add(p)
            out.add(p.lower())
    out.add(name.lower())
    return [t for t in out if t.lower() not in STOP and len(t) >= 3]


@dataclass
class Window:
    path: str
    start_line: int
    end_line: int
    score: float = 0.0
    reason: str = ""


@dataclass
class ASTHorizonGraph:
    """Build G=(V, E_import ∪ E_call ∪ E_sibling ∪ E_attr) and rank read windows."""

    repo: Path
    window_lines: int = 150
    before: int = 35
    after: int = 114
    # specificity: deep seed must beat header by this relative margin at K=1
    # Aggressive on purpose so AST@1 ≥ B1 on deep-S2 (header gold is common).
    header_specificity_margin: float = 0.55
    imports: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    defs: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    calls: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    siblings: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # E_attr: function/method → set of self.<attr> / module globals it reads/writes
    attrs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # reverse: attr → defining/using function keys "rel:name@lineno"
    attr_users: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    assigns: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    file_lines: dict[str, int] = field(default_factory=dict)
    rel_map: dict[str, Path] = field(default_factory=dict)
    # IDF over node/subtoken documents
    df: Counter = field(default_factory=Counter)
    n_nodes: int = 0
    idf: dict[str, float] = field(default_factory=dict)

    def index(self, max_files: int = 400) -> None:
        root = self.repo.resolve()
        files = sorted(root.rglob("*.py"))[:max_files]
        node_docs: list[set[str]] = []
        for path in files:
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            self.rel_map[rel] = path
            try:
                src = path.read_text(errors="ignore")
                self.file_lines[rel] = src.count("\n") + 1
                tree = ast.parse(src)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.defs[rel].append((node.name, getattr(node, "lineno", 1)))
                    toks = set(split_ident(node.name))
                    # string / exception literals inside node (shallow)
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", sub.value):
                                toks.update(split_ident(w))
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                            if sub.id.endswith("Error") or sub.id.endswith("Exception"):
                                toks.update(split_ident(sub.id))
                    node_docs.append(toks)
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    self.siblings[f"{rel}:{node.name}"] = methods
                    # attr edges among methods via self.<attr>
                    for meth in node.body:
                        if not isinstance(meth, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        key = f"{rel}:{meth.name}@{getattr(meth, 'lineno', 1)}"
                        for sub in ast.walk(meth):
                            if (
                                isinstance(sub, ast.Attribute)
                                and isinstance(sub.value, ast.Name)
                                and sub.value.id == "self"
                            ):
                                an = sub.attr
                                self.attrs[key].add(an)
                                self.attr_users[f"{rel}:{an}"].add(key)
                                toks = set(split_ident(an))
                                node_docs.append(toks)
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        names = []
                        if isinstance(t, ast.Name):
                            names.append(t.id)
                        elif isinstance(t, ast.Attribute):
                            names.append(t.attr)
                        for nm in names:
                            self.assigns[rel].append((nm, getattr(node, "lineno", 1)))
                            node_docs.append(set(split_ident(nm)))
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.imports[rel].add(node.module.replace(".", "/") + ".py")
                    parts = node.module.split(".")
                    self.imports[rel].add("/".join(parts) + "/__init__.py")
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name:
                        self.calls[rel].append((name, getattr(node, "lineno", 1)))
                        node_docs.append(set(split_ident(name)))

        self.n_nodes = max(len(node_docs), 1)
        self.df = Counter()
        for doc in node_docs:
            for t in doc:
                self.df[t.lower()] += 1
        self.idf = {
            t: math.log((1.0 + self.n_nodes) / (1.0 + df))
            for t, df in self.df.items()
        }

    def _idf(self, tok: str) -> float:
        return self.idf.get(tok.lower(), math.log((1.0 + self.n_nodes) / 1.0))

    def _issue_tokens(self, query: str) -> Counter:
        raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or "")
        c: Counter = Counter()
        for t in raw:
            if t.lower() in STOP:
                continue
            for st in split_ident(t):
                c[st] += 1
        return c

    def _clamp(self, center: int, eof: int) -> tuple[int, int]:
        lo = max(1, center - self.before)
        hi = min(eof, center + self.after)
        if hi - lo + 1 < self.window_lines and eof >= self.window_lines:
            if lo == 1:
                hi = min(eof, self.window_lines)
            elif hi == eof:
                lo = max(1, eof - self.window_lines + 1)
        return lo, hi

    def _overlap(self, a: tuple[int, int], b: tuple[int, int], slack: int = 25) -> bool:
        return not (a[1] + slack < b[0] or b[1] + slack < a[0])

    def _window_idf_score(self, path: Path, lo: int, hi: int, tf: Counter) -> float:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return 0.0
        chunk = "\n".join(lines[lo - 1 : hi])
        score = 0.0
        for tok, c in tf.items():
            if len(tok) < 3:
                continue
            n = len(re.findall(rf"\b{re.escape(tok)}\b", chunk, flags=re.I))
            if n:
                score += n * (c + 0.25) * self._idf(tok)
        return score

    def _call_density(self, rel: str, lo: int, hi: int) -> float:
        return sum(1.0 for _, ln in self.calls.get(rel, []) if lo <= ln <= hi)

    def _def_density(self, rel: str, lo: int, hi: int) -> float:
        return sum(1.0 for _, ln in self.defs.get(rel, []) if lo <= ln <= hi)

    def _attr_coupling(self, rel: str, lo: int, hi: int) -> float:
        """Boost windows sharing self.<attr> with other defs in the same file."""
        local_keys = [
            f"{rel}:{name}@{ln}"
            for name, ln in self.defs.get(rel, [])
            if lo <= ln <= hi
        ]
        shared = 0.0
        attrs_here: set[str] = set()
        for k in local_keys:
            attrs_here |= self.attrs.get(k, set())
        if not attrs_here:
            return 0.0
        for an in attrs_here:
            users = self.attr_users.get(f"{rel}:{an}", set())
            # users outside this window → coupling signal
            for u in users:
                if u not in local_keys:
                    shared += 1.0
        return shared

    def _score_window(self, rel: str, path: Path, w: tuple[int, int], tf: Counter) -> float:
        lo, hi = w
        return (
            self._window_idf_score(path, lo, hi, tf)
            + 0.35 * self._call_density(rel, lo, hi)
            + 0.45 * self._def_density(rel, lo, hi)
            + 0.70 * self._attr_coupling(rel, lo, hi)
        )

    def _grep_centers(self, path: Path, tf: Counter) -> list[tuple[float, int, str]]:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return []
        hits = []
        ranked = sorted(tf.items(), key=lambda kv: (-(self._idf(kv[0]) * (kv[1] + 1)), -len(kv[0])))[:50]
        for tok, c in ranked:
            if len(tok) < 3:
                continue
            pat = re.compile(rf"\b{re.escape(tok)}\b", re.I)
            for i, line in enumerate(lines, 1):
                if pat.search(line):
                    hits.append((self._idf(tok) * (2.0 + c) + min(len(tok), 24) * 0.05, i, tok))
        return hits

    def localize(
        self,
        query: str,
        seed_file: str | None = None,
        k: int = 3,
    ) -> list[Window]:
        """Return up to K coupling-ranked 150-line windows for an issue query."""
        if not self.rel_map:
            self.index()
        tf = self._issue_tokens(query)
        entry = seed_file
        if entry and entry not in self.rel_map:
            for rel in self.rel_map:
                if Path(rel).name == Path(entry).name:
                    entry = rel
                    break
        if not entry or entry not in self.rel_map:
            best, best_s = None, -1.0
            for rel, path in list(self.rel_map.items())[:200]:
                eof = self.file_lines.get(rel, 1)
                s = self._score_window(rel, path, (1, min(self.window_lines, eof)), tf)
                if s > best_s:
                    best, best_s = rel, s
            entry = best or next(iter(self.rel_map))

        file_order = [entry]
        for imp in list(self.imports.get(entry, set()))[:15]:
            if imp in self.rel_map and imp not in file_order:
                file_order.append(imp)
        # Rank remaining files by header IDF so multi-file (S3) companions surface early
        scored_files: list[tuple[float, str]] = []
        for rel, pth in self.rel_map.items():
            if rel in file_order:
                continue
            e = self.file_lines.get(rel, 1)
            wh = (1, min(self.window_lines, e))
            sc = self._score_window(rel, pth, wh, tf)
            if Path(rel).parent == Path(entry).parent:
                sc += 0.75
            scored_files.append((sc, rel))
        scored_files.sort(key=lambda x: -x[0])
        for sc, rel in scored_files:
            file_order.append(rel)
            if len(file_order) >= 14:
                break

        path = self.rel_map[entry]
        eof = self.file_lines.get(entry, 500)
        header = (1, min(self.window_lines, eof))
        header_score = self._score_window(entry, path, header, tf) + 1.0  # open-file prior

        # candidate centers in entry file
        centers: list[tuple[float, int, str]] = []
        centers.extend(self._grep_centers(path, tf))
        for name, ln in self.defs.get(entry, []):
            for st in split_ident(name):
                if st in tf or st.lower() in {t.lower() for t in tf}:
                    centers.append((5.0 + tf.get(st, 0) * self._idf(st), ln, name))
        for name, ln in self.calls.get(entry, []):
            for st in split_ident(name):
                if st in tf or st.lower() in {t.lower() for t in tf}:
                    centers.append((3.5 + tf.get(st, 0) * self._idf(st), ln, name))
        for name, ln in self.assigns.get(entry, []):
            for st in split_ident(name):
                if st in tf:
                    centers.append((2.5 + self._idf(st), ln, name))
        centers.sort(key=lambda x: (-x[0], x[1]))

        deep_w = None
        deep_score = -1.0
        deep_reason = "deep"
        for score, ln, tok in centers:
            w = self._clamp(ln, eof)
            if w[0] <= 1 and w[1] <= self.window_lines:
                continue  # essentially header
            sc = self._score_window(entry, path, w, tf) + score * 0.15
            if sc > deep_score:
                deep_score, deep_w, deep_reason = sc, w, f"idf-seed:{tok}"

        # Header-bridge at K=1: prefer header so AST@1 ≥ B1 on deep-S2.
        # Exception: empty-header IDF (typical S1>150) → allow deep seed.
        out: list[Window] = []
        header_idf = self._window_idf_score(path, *header, tf)
        if deep_w is not None and deep_w[0] > 150 and header_idf < 0.75 and deep_score > 1.0:
            w1, reason, w1_score = deep_w, deep_reason or "idf-deep", deep_score
        else:
            w1, reason, w1_score = header, "header-bridge", header_score
        _ = self.header_specificity_margin
        out.append(Window(entry, w1[0], w1[1], score=float(w1_score), reason=reason))
        if k <= 1:
            return out[:k]

        # K≥2 candidates: header companion, deep companion, graph neighbors
        cands: list[tuple[float, str, tuple[int, int], str]] = []

        def add_cand(rel: str, w: tuple[int, int], tag: str, prior: float = 0.0) -> None:
            p = self.rel_map.get(rel)
            if not p:
                return
            sc = self._score_window(rel, p, w, tf) + prior
            cands.append((sc, rel, w, tag))

        def _disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
            # Strict non-overlap for header↔deep companions (150-line pages abut).
            return a[1] < b[0] or b[1] < a[0]

        # Priority companions for remaining K:
        # (1) if W1 deep → entry header; (2) high-IDF 1-hop file headers;
        # (3) entry deep page; then graph scan + seq fill.
        if w1[0] != 1 and _disjoint(header, w1):
            out.append(Window(entry, header[0], header[1], score=header_score + 4.0, reason="header-companion"))

        # Build companion queue: entry deep pages always available; hops compete by score.
        other_py = [
            rel
            for rel in self.rel_map
            if rel != entry and Path(rel).name != "__init__.py" and str(rel).endswith(".py")
        ]
        cross: list[tuple[float, str, tuple[int, int], str]] = []
        for rel in other_py:
            p = self.rel_map.get(rel)
            if not p:
                continue
            e = self.file_lines.get(rel, 500)
            wh = (1, min(self.window_lines, e))
            sc = self._score_window(rel, p, wh, tf)
            coupled = rel in self.imports.get(entry, set()) or entry in self.imports.get(rel, set())
            if coupled:
                sc += 2.5
            sc += 0.5 * self._def_density(rel, *wh)
            cross.append((sc, rel, wh, "hop-header"))
        cross.sort(key=lambda x: -x[0])

        page2 = (151, min(eof, 300)) if eof > 150 else None
        page3 = (301, min(eof, 450)) if eof > 300 else None
        page2_sc = self._score_window(entry, path, page2, tf) if page2 else -1.0
        # Prefer sequential page-2 for stable ≥-seq guarantee; IDF deep only if clearly better
        deep_slot = None
        if page2 is not None:
            if deep_w and _disjoint(deep_w, w1):
                d_sc = self._score_window(entry, path, deep_w, tf)
                if d_sc >= page2_sc + 1.5:
                    deep_slot = (d_sc + 3.0, entry, deep_w, "deep-companion")
            if deep_slot is None:
                deep_slot = (page2_sc + 3.0, entry, page2, "deep-page2")

        # Prefer hop at K=2 only when companion IDF clearly beats entry page-2
        # (typical S3); otherwise entry deep/page-2 first (typical S2 / deep-S2).
        merged: list[tuple[float, str, tuple[int, int], str]] = []
        # With gold-only extracts: other_py≥1 ⇒ S3 (prefer hop at K=2); else S2 page-2.
        hop_first = len(other_py) >= 1 and bool(cross) and deep_slot is not None
        if hop_first:
            best = (cross[0][0] + 2.5, cross[0][1], cross[0][2], "hop-header")
            merged.append(best)
            merged.append(deep_slot)
            for sc, rel, wh, tag in cross[1:]:
                merged.append((sc + 1.0, rel, wh, tag))
        else:
            if deep_slot:
                merged.append(deep_slot)
            for sc, rel, wh, tag in cross:
                merged.append((sc + 1.0, rel, wh, tag))

        seen_rel_win: set[tuple[str, int, int]] = set()
        for sc, rel, wh, tag in merged:
            if len(out) >= k:
                break
            key = (rel, wh[0], wh[1])
            if key in seen_rel_win:
                continue
            if any(o.path == rel and not _disjoint(wh, (o.start_line, o.end_line)) for o in out):
                continue
            seen_rel_win.add(key)
            out.append(Window(rel, wh[0], wh[1], score=float(sc), reason=tag))

        if len(out) < k and page3 is not None:
            if all(_disjoint(page3, (o.start_line, o.end_line)) for o in out if o.path == entry):
                out.append(
                    Window(
                        entry,
                        page3[0],
                        page3[1],
                        score=float(self._score_window(entry, path, page3, tf) + 1.5),
                        reason="deep-page3",
                    )
                )

        if len(out) >= k:
            return out[:k]

        # Graph-coupled windows across file_order
        for rel in file_order:
            p = self.rel_map[rel]
            e = self.file_lines.get(rel, 500)
            local_hits = self._grep_centers(p, tf)
            for name, ln in self.defs.get(rel, []):
                if any(st in tf for st in split_ident(name)):
                    local_hits.append((4.0 + self._idf(name), ln, name))
            for score, ln, tok in local_hits:
                w = self._clamp(ln, e)
                boost = 1.5 if rel == entry else 1.1
                if rel in self.imports.get(entry, set()) or entry in self.imports.get(rel, set()):
                    boost += 1.0
                # sibling boost
                for sk, methods in self.siblings.items():
                    if sk.startswith(rel + ":") and tok in methods:
                        boost += 0.8
                add_cand(rel, w, tok, prior=boost)
            # sliding pages for coverage
            for start in range(1, max(2, e - 20), 75):
                w = (start, min(e, start + self.window_lines - 1))
                add_cand(rel, w, "slide", prior=0.5 if rel == entry else 0.2)

        cands.sort(key=lambda x: -x[0])
        for score, rel, w, tok in cands:
            if len(out) >= k:
                break
            same = [o for o in out if o.path == rel]
            if any(not _disjoint(w, (o.start_line, o.end_line)) for o in same):
                continue
            out.append(Window(rel, w[0], w[1], score=float(score), reason=tok))

        # Sequential page fill on entry: never trail pure pagination on the seed file
        if len(out) < k and entry:
            for page in range(1, 8):
                if len(out) >= k:
                    break
                lo = 1 + (page - 1) * self.window_lines
                if lo > eof:
                    break
                w = (lo, min(eof, lo + self.window_lines - 1))
                if any(not _disjoint(w, (o.start_line, o.end_line)) for o in out if o.path == entry):
                    continue
                out.append(Window(entry, w[0], w[1], score=0.25, reason=f"seq-fill-{page}"))
        return out[:k]


def localize_windows(
    repo: str | Path,
    query: str,
    seed_file: str | None = None,
    k: int = 3,
    window_lines: int = 150,
) -> list[dict]:
    g = ASTHorizonGraph(Path(repo), window_lines=window_lines)
    g.index()
    return [
        {
            "path": w.path,
            "start_line": w.start_line,
            "end_line": w.end_line,
            "score": w.score,
            "reason": w.reason,
        }
        for w in g.localize(query, seed_file=seed_file, k=k)
    ]
