from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Tuple

WILDCARD = None  # internal marker


def parse_hex_bytes(hex_str: str) -> bytes:
    hex_str = hex_str.strip()
    if not hex_str:
        return b""
    parts = hex_str.split()
    return bytes(int(p, 16) for p in parts)


def fmt_pattern(pat: Tuple[int | None, ...]) -> str:
    # Render wildcarded pattern as "aa ?? f2 de"
    out = []
    for x in pat:
        out.append("??" if x is None else f"{x:02x}")
    return " ".join(out)


def all_kgrams(seq: bytes, k: int) -> Iterable[bytes]:
    if len(seq) < k:
        return
    for i in range(0, len(seq) - k + 1):
        yield seq[i:i + k]


def mask_pattern(kgram: bytes, mask_positions: Tuple[int, ...]) -> Tuple[int | None, ...]:
    # Convert to tuple of ints / None so it’s hashable
    t = list(kgram)
    for pos in mask_positions:
        t[pos] = WILDCARD
    return tuple(t)


def generate_masks(k: int, max_wildcards: int) -> List[Tuple[int, ...]]:
    """
    Returns all position-combinations of size 0..max_wildcards.
    NOTE: This grows as sum_{w=0..W} C(k,w). Keep W small (1-3).
    """
    masks: List[Tuple[int, ...]] = []
    for w in range(0, max_wildcards + 1):
        masks.extend(combinations(range(k), w))
    return masks


def generate_contiguous_masks(k: int, max_wildcards: int) -> List[Tuple[int, ...]]:
    """
    Contiguous wildcard runs are often good for immediates (e.g., 2–4 bytes).
    This generates masks like (i,i+1,i+2) up to length max_wildcards.
    """
    masks: List[Tuple[int, ...]] = [tuple()]  # include no-mask
    for w in range(1, max_wildcards + 1):
        for start in range(0, k - w + 1):
            masks.append(tuple(range(start, start + w)))
    return masks


def mine_wildcards(
    windows: List[bytes],
    k: int,
    max_wildcards: int,
    contiguous_only: bool,
    min_count: int,
) -> Counter:
    masks = generate_contiguous_masks(k, max_wildcards) if contiguous_only else generate_masks(k, max_wildcards)
    counts = Counter()

    for w in windows:
        for kg in all_kgrams(w, k):
            for m in masks:
                counts[mask_pattern(kg, m)] += 1

    # Filter tiny counts
    if min_count > 1:
        counts = Counter({p: c for p, c in counts.items() if c >= min_count})

    return counts


def load_windows(boundary_json: Path, which: str) -> List[bytes]:
    data = json.loads(boundary_json.read_text(encoding="utf-8"))
    out: List[bytes] = []
    for cand in data.get("candidates", []):
        wh = cand.get("windows_hex", {}).get(which, "")
        out.append(parse_hex_bytes(wh))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="boundary_patterns.json")
    ap.add_argument("--which", choices=["start_post", "end_pre", "start_pre", "end_post"], default="start_post")
    ap.add_argument("--k", type=int, default=8, help="k-gram size (try 6,8,10,12)")
    ap.add_argument("--W", type=int, default=2, help="max wildcard bytes per pattern (keep small: 1-3)")
    ap.add_argument("--contiguous", action="store_true", help="only allow contiguous wildcard runs (good for immediates)")
    ap.add_argument("--min-count", type=int, default=6, help="only report patterns with >= this count")
    ap.add_argument("--top", type=int, default=30, help="top patterns to print")
    args = ap.parse_args()

    windows = load_windows(Path(args.inp), args.which)

    counts = mine_wildcards(
        windows=windows,
        k=args.k,
        max_wildcards=args.W,
        contiguous_only=args.contiguous,
        min_count=args.min_count,
    )

    # Rank: count desc, then fewer wildcards (more specific), then length (same k anyway)
    def key_fn(item):
        pat, c = item
        wc = sum(1 for x in pat if x is None)
        return (c, -wc)

    ranked = sorted(counts.items(), key=key_fn, reverse=True)[:args.top]

    print(f"Window: {args.which} | k={args.k} | max_wildcards={args.W} | contiguous_only={args.contiguous}")
    print(f"Candidates: {len(windows)}")
    print()

    for pat, c in ranked:
        print(f"{c:>6}  {fmt_pattern(pat)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
