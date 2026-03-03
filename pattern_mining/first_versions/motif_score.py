#!/usr/bin/env python3
"""
motif_score.py

ISA-agnostic boundary scoring using wildcard motifs.

What it does:
1) Loads boundary_patterns.json (produced by your boundary mining script).
2) Mines wildcard motifs separately in:
   - start_post  (prologue-ish area)
   - end_pre     (epilogue-ish area)
3) Scores each candidate by scanning its window for occurrences of the mined motifs.
4) Writes:
   - CSV ranking candidates by score
   - JSON with motifs + per-candidate hit details

Example:
  python motif_score.py --in report_out/boundary_patterns.json --out scored \
    --k-start 8 --W-start 3 --min-count-start 4 --top-start 40 --contig-start \
    --k-end 8 --W-end 3 --min-count-end 4 --top-end 40 --contig-end

Notes:
- Wildcards are contiguous runs by default when --contig-* is set, which is good for immediates.
- If you think variation is scattered (register fields), try without --contig-* and smaller W (like 2).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

WILDCARD = None  # internal marker for "??"


# ----------------------------
# Parsing helpers
# ----------------------------

def parse_hex_bytes(hex_str: str) -> bytes:
    hex_str = (hex_str or "").strip()
    if not hex_str:
        return b""
    return bytes(int(p, 16) for p in hex_str.split())


def fmt_pattern(pat: Tuple[int | None, ...]) -> str:
    return " ".join("??" if x is None else f"{x:02x}" for x in pat)


def fixed_count(pat: Tuple[int | None, ...]) -> int:
    return sum(1 for x in pat if x is not None)


def all_kgrams(seq: bytes, k: int) -> Iterable[Tuple[int, bytes]]:
    """Yield (offset, kgram_bytes)."""
    if len(seq) < k:
        return
    for i in range(0, len(seq) - k + 1):
        yield i, seq[i:i + k]


def mask_pattern(kgram: bytes, mask_positions: Tuple[int, ...]) -> Tuple[int | None, ...]:
    t = list(kgram)
    for pos in mask_positions:
        t[pos] = WILDCARD
    return tuple(t)


def generate_masks(k: int, max_wildcards: int) -> List[Tuple[int, ...]]:
    masks: List[Tuple[int, ...]] = []
    for w in range(0, max_wildcards + 1):
        masks.extend(combinations(range(k), w))
    return masks


def generate_contiguous_masks(k: int, max_wildcards: int) -> List[Tuple[int, ...]]:
    masks: List[Tuple[int, ...]] = [tuple()]  # no-mask included
    for w in range(1, max_wildcards + 1):
        for start in range(0, k - w + 1):
            masks.append(tuple(range(start, start + w)))
    return masks


# ----------------------------
# Motif mining
# ----------------------------

@dataclass(frozen=True)
class Motif:
    pattern: Tuple[int | None, ...]
    count: int

    @property
    def k(self) -> int:
        return len(self.pattern)

    @property
    def fixed(self) -> int:
        return fixed_count(self.pattern)

    def to_json(self) -> Dict:
        return {
            "pattern": fmt_pattern(self.pattern),
            "k": self.k,
            "fixed": self.fixed,
            "count": self.count,
        }


def mine_wildcard_motifs(
    windows: List[bytes],
    k: int,
    max_wildcards: int,
    contiguous_only: bool,
    min_count: int,
    top_n: int,
) -> List[Motif]:
    masks = generate_contiguous_masks(k, max_wildcards) if contiguous_only else generate_masks(k, max_wildcards)
    counts = Counter()

    for w in windows:
        for _, kg in all_kgrams(w, k):
            for m in masks:
                counts[mask_pattern(kg, m)] += 1

    # filter
    items = [(p, c) for p, c in counts.items() if c >= min_count]

    # rank: count desc, then more fixed bytes (more specific), then fewer wildcards
    # (fixed desc == wildcards asc since k is constant)
    items.sort(key=lambda pc: (pc[1], fixed_count(pc[0])), reverse=True)

    motifs = [Motif(pattern=p, count=c) for p, c in items[:top_n]]
    return motifs


# ----------------------------
# Matching + scoring
# ----------------------------

@dataclass
class MotifHit:
    motif_index: int
    motif_pattern: str
    motif_count: int
    motif_fixed: int
    offset: int  # offset within the window where match starts
    matched_kgram: str  # concrete bytes from window at that position
    weight: float


def motif_matches_at(window: bytes, pat: Tuple[int | None, ...], offset: int) -> bool:
    k = len(pat)
    if offset < 0 or offset + k > len(window):
        return False
    seg = window[offset:offset + k]
    for i, pv in enumerate(pat):
        if pv is None:
            continue
        if seg[i] != pv:
            return False
    return True


def find_first_match(window: bytes, pat: Tuple[int | None, ...]) -> Optional[int]:
    k = len(pat)
    if len(window) < k:
        return None
    for off in range(0, len(window) - k + 1):
        if motif_matches_at(window, pat, off):
            return off
    return None


def weight_motif(m: Motif, alpha_k: float, alpha_fixed: float, alpha_count: float) -> float:
    """
    Simple, interpretable weighting:
      weight = (count^alpha_count) * (k^alpha_k) * (fixed^alpha_fixed)

    Defaults give sensible behavior:
      - higher motif frequency matters
      - longer motifs matter
      - more fixed bytes (more specific) matter
    """
    return (m.count ** alpha_count) * (m.k ** alpha_k) * (m.fixed ** alpha_fixed)


def score_window(
    window: bytes,
    motifs: List[Motif],
    alpha_k: float,
    alpha_fixed: float,
    alpha_count: float,
    max_hits: int,
) -> Tuple[float, List[MotifHit]]:
    total = 0.0
    hits: List[MotifHit] = []

    # Add each motif at most once per window (avoid double-counting sliding overlaps)
    for idx, m in enumerate(motifs):
        off = find_first_match(window, m.pattern)
        if off is None:
            continue

        w = weight_motif(m, alpha_k=alpha_k, alpha_fixed=alpha_fixed, alpha_count=alpha_count)
        total += w

        seg = window[off:off + m.k]
        hits.append(MotifHit(
            motif_index=idx,
            motif_pattern=fmt_pattern(m.pattern),
            motif_count=m.count,
            motif_fixed=m.fixed,
            offset=off,
            matched_kgram=" ".join(f"{b:02x}" for b in seg),
            weight=w,
        ))

    # Keep only top hits for readability
    hits.sort(key=lambda h: h.weight, reverse=True)
    if max_hits > 0:
        hits = hits[:max_hits]

    return total, hits


# ----------------------------
# Main
# ----------------------------

def load_candidates(boundary_json: Path) -> Dict:
    return json.loads(boundary_json.read_text(encoding="utf-8"))


def extract_windows(data: Dict, which: str) -> List[bytes]:
    out: List[bytes] = []
    for cand in data.get("candidates", []):
        hex_w = cand.get("windows_hex", {}).get(which, "")
        out.append(parse_hex_bytes(hex_w))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="boundary_patterns.json")
    ap.add_argument("--out", required=True, help="output directory")

    # Motif mining parameters (start_post)
    ap.add_argument("--k-start", type=int, default=8)
    ap.add_argument("--W-start", type=int, default=3)
    ap.add_argument("--min-count-start", type=int, default=4)
    ap.add_argument("--top-start", type=int, default=40)
    ap.add_argument("--contig-start", action="store_true", help="use contiguous wildcards for start_post")

    # Motif mining parameters (end_pre)
    ap.add_argument("--k-end", type=int, default=8)
    ap.add_argument("--W-end", type=int, default=3)
    ap.add_argument("--min-count-end", type=int, default=4)
    ap.add_argument("--top-end", type=int, default=40)
    ap.add_argument("--contig-end", action="store_true", help="use contiguous wildcards for end_pre")

    # Scoring weights
    ap.add_argument("--alpha-k", type=float, default=1.0)
    ap.add_argument("--alpha-fixed", type=float, default=1.0)
    ap.add_argument("--alpha-count", type=float, default=1.0)

    ap.add_argument("--max-hits-per-side", type=int, default=10, help="store at most N hits per side in JSON")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_candidates(inp)

    # Extract windows
    start_post_windows = extract_windows(data, "start_post")
    end_pre_windows = extract_windows(data, "end_pre")

    # Mine motifs
    start_motifs = mine_wildcard_motifs(
        windows=start_post_windows,
        k=args.k_start,
        max_wildcards=args.W_start,
        contiguous_only=args.contig_start,
        min_count=args.min_count_start,
        top_n=args.top_start,
    )
    end_motifs = mine_wildcard_motifs(
        windows=end_pre_windows,
        k=args.k_end,
        max_wildcards=args.W_end,
        contiguous_only=args.contig_end,
        min_count=args.min_count_end,
        top_n=args.top_end,
    )

    # Score candidates
    scored = []
    candidates = data.get("candidates", [])
    for i, cand in enumerate(candidates):
        sp = parse_hex_bytes(cand.get("windows_hex", {}).get("start_post", ""))
        ep = parse_hex_bytes(cand.get("windows_hex", {}).get("end_pre", ""))

        start_score, start_hits = score_window(
            sp, start_motifs,
            alpha_k=args.alpha_k, alpha_fixed=args.alpha_fixed, alpha_count=args.alpha_count,
            max_hits=args.max_hits_per_side
        )
        end_score, end_hits = score_window(
            ep, end_motifs,
            alpha_k=args.alpha_k, alpha_fixed=args.alpha_fixed, alpha_count=args.alpha_count,
            max_hits=args.max_hits_per_side
        )

        total = start_score + end_score

        scored.append({
            "idx": i,
            "base_file": cand.get("base_file"),
            "new_file": cand.get("new_file"),
            "side_file": cand.get("side_file"),
            "start": cand.get("start"),
            "end": cand.get("end"),
            "note": cand.get("note"),
            "start_score": start_score,
            "end_score": end_score,
            "total_score": total,
            "start_hits": [h.__dict__ for h in start_hits],
            "end_hits": [h.__dict__ for h in end_hits],
        })

    scored.sort(key=lambda r: r["total_score"], reverse=True)

    # Write CSV (easy to inspect)
    csv_path = out_dir / "candidate_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "idx", "total_score", "start_score", "end_score",
            "base_file", "new_file", "side_file", "start", "end", "note",
            "top_start_hit", "top_end_hit"
        ])
        for rank, r in enumerate(scored, start=1):
            top_s = r["start_hits"][0]["motif_pattern"] if r["start_hits"] else ""
            top_e = r["end_hits"][0]["motif_pattern"] if r["end_hits"] else ""
            w.writerow([
                rank, r["idx"],
                f"{r['total_score']:.3f}", f"{r['start_score']:.3f}", f"{r['end_score']:.3f}",
                r["base_file"], r["new_file"], r["side_file"], r["start"], r["end"], r["note"],
                top_s, top_e
            ])

    # Write JSON with full details + motifs
    json_path = out_dir / "candidate_scores.json"
    payload = {
        "meta": {
            "input": str(inp),
            "num_candidates": len(candidates),
            "scoring": {
                "alpha_k": args.alpha_k,
                "alpha_fixed": args.alpha_fixed,
                "alpha_count": args.alpha_count,
                "max_hits_per_side": args.max_hits_per_side,
            },
            "motif_mining": {
                "start_post": {
                    "k": args.k_start, "W": args.W_start, "min_count": args.min_count_start,
                    "top": args.top_start, "contiguous_only": args.contig_start
                },
                "end_pre": {
                    "k": args.k_end, "W": args.W_end, "min_count": args.min_count_end,
                    "top": args.top_end, "contiguous_only": args.contig_end
                },
            },
        },
        "motifs": {
            "start_post": [m.to_json() for m in start_motifs],
            "end_pre": [m.to_json() for m in end_motifs],
        },
        "candidates_ranked": scored,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Print quick summary
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print()
    print("Top start_post motifs:")
    for m in start_motifs[:10]:
        print(f"{m.count:>4}  {fmt_pattern(m.pattern)}")
    print()
    print("Top end_pre motifs:")
    for m in end_motifs[:10]:
        print(f"{m.count:>4}  {fmt_pattern(m.pattern)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
