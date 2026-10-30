"""AST-HorizonGraph: 1-hop import/call/sibling localization under a read-file horizon."""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Window:
    path: str
    start_line: int
    end_line: int
    score: float = 0.0
    reason: str = ""


@dataclass
class ASTHorizonGraph:
    """Build G=(V, E_import ∪ E_call ∪ E_sibling) over a Python repo and rank read windows."""

    repo: Path
    window_lines: int = 150
    before: int = 35
    after: int = 114
    imports: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    defs: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    calls: dict[str, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    siblings: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    file_lines: dict[str, int] = field(default_factory=dict)
    rel_map: dict[str, Path] = field(default_factory=dict)

    def index(self, max_files: int = 400) -> None:
        root = self.repo.resolve()
        files = sorted(root.rglob("*.py"))[:max_files]
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
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    self.siblings[f"{rel}:{node.name}"] = methods
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

    def _issue_tokens(self, query: str) -> Counter:
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or "")
        stop = {
            "the", "and", "for", "that", "this", "with", "from", "when", "have",
            "should", "would", "could", "into", "about", "test", "tests", "assert",
            "error", "exception", "return", "true", "false", "none", "self", "class",
            "def", "import", "file", "code", "issue", "bug", "fix", "please",
        }
        return Counter(t for t in toks if t.lower() not in stop)

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

    def _density(self, path: Path, lo: int, hi: int, tf: Counter) -> float:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return 0.0
        chunk = "\n".join(lines[lo - 1 : hi])
        return sum(chunk.count(tok) * (c + len(tok) * 0.05) for tok, c in tf.items() if len(tok) >= 4)

    def _grep_hits(self, path: Path, tf: Counter) -> list[tuple[float, int, str]]:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return []
        hits = []
        for tok, c in sorted(tf.items(), key=lambda kv: (-(len(kv[0]) + kv[1]), -len(kv[0])))[:40]:
            if len(tok) < 4:
                continue
            pat = re.compile(rf"\b{re.escape(tok)}\b")
            for i, line in enumerate(lines, 1):
                if pat.search(line):
                    hits.append((3.0 + c + min(len(tok), 20) * 0.1, i, tok))
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
        # choose entry file
        entry = seed_file
        if entry and entry not in self.rel_map:
            # basename fallback
            for rel in self.rel_map:
                if Path(rel).name == Path(entry).name:
                    entry = rel
                    break
        if not entry or entry not in self.rel_map:
            # pick file with densest issue tokens
            best, best_s = None, -1.0
            for rel, path in list(self.rel_map.items())[:200]:
                eof = self.file_lines.get(rel, 1)
                s = self._density(path, 1, min(self.window_lines, eof), tf)
                if s > best_s:
                    best, best_s = rel, s
            entry = best or next(iter(self.rel_map))

        file_order = [entry]
        for imp in list(self.imports.get(entry, set()))[:15]:
            if imp in self.rel_map and imp not in file_order:
                file_order.append(imp)
        for rel in self.rel_map:
            if rel not in file_order and Path(rel).parent == Path(entry).parent:
                file_order.append(rel)
            if len(file_order) >= 12:
                break

        # Window 1: entry only
        out: list[Window] = []
        path = self.rel_map[entry]
        eof = self.file_lines.get(entry, 500)
        header = (1, min(self.window_lines, eof))
        hits = self._grep_hits(path, tf)
        for name, ln in self.defs.get(entry, []):
            if name in tf:
                hits.append((5.0 + tf[name], ln, name))
        hits.sort(key=lambda x: (-x[0], x[1]))
        if hits and hits[0][1] > 115:
            deep = self._clamp(hits[0][1], eof)
            if self._density(path, *deep, tf) >= self._density(path, *header, tf) * 0.85:
                w1 = deep
                reason = f"issue-seed:{hits[0][2]}"
            else:
                w1, reason = header, "header"
        else:
            w1, reason = header, "header"
        out.append(Window(entry, w1[0], w1[1], score=1.0, reason=reason))

        # Remaining: score candidates across graph
        cands: list[tuple[float, str, tuple[int, int], str]] = []
        for rel in file_order:
            p = self.rel_map[rel]
            e = self.file_lines.get(rel, 500)
            local_hits = self._grep_hits(p, tf)
            for name, ln in self.defs.get(rel, []):
                if name in tf or any(name == h[2] for h in local_hits[:5]):
                    local_hits.append((4.0 + tf.get(name, 0), ln, name))
            for score, ln, tok in local_hits:
                w = self._clamp(ln, e)
                dens = self._density(p, *w, tf)
                boost = 1.5 if rel == entry else 1.2
                if rel in self.imports.get(entry, set()) or entry in self.imports.get(rel, set()):
                    boost += 0.8
                cands.append((score + dens * boost, rel, w, tok))
            # sliding pages
            for start in range(1, max(2, e - 20), 75):
                w = (start, min(e, start + self.window_lines - 1))
                dens = self._density(p, *w, tf)
                cands.append((dens * (1.4 if rel == entry else 1.0), rel, w, "slide"))

        cands.sort(key=lambda x: -x[0])
        for score, rel, w, tok in cands:
            if len(out) >= k:
                break
            same = [o for o in out if o.path == rel]
            if any(self._overlap(w, (o.start_line, o.end_line)) for o in same):
                continue
            if rel == entry and self._overlap(w, (out[0].start_line, out[0].end_line), slack=40):
                continue
            out.append(Window(rel, w[0], w[1], score=float(score), reason=tok))
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
