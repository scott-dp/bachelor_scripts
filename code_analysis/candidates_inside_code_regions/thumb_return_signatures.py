#!/usr/bin/env python3
"""
thumb_return_signatures.py

Selects "code-like" regions using the SAME block-level Thumb heuristic as
thumb_code_only_histograms.py, then counts very specific Thumb return/prologue
signatures inside those kept regions.

Outputs a CSV with per-device rates (per 1k halfwords) for:
- POP {...,PC}   : 0xBDxx
- BX LR          : 0x4770
- MOV PC, LR     : 0x46F7 (common Thumb16 encoding)
- PUSH {...,LR}  : 0xB5xx

IMPORTANT:
- Halfwords are read little-endian (bytes 70 47 => 0x4770).
- 32-bit Thumb-2 instructions are handled like the histogram script:
  if a valid 32-bit pair is detected, we mark ONLY the first halfword as an
  instruction start for validity ratio purposes.

Usage example (matching your earlier runs):
  python thumb_return_signatures.py --bin-root ../../../bin --offset 6001 --block-bytes 2048 --min-ratio 0.9 --out out_code
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def u16_le(data: bytes, i: int) -> int:
    return data[i] | (data[i + 1] << 8)


def is_thumb16(h: int) -> bool:
    """
    Heuristic recognizer for 16-bit Thumb (covers most common encodings).

    IMPORTANT: This is intentionally permissive (we want "likely code blocks",
    not perfect disassembly). The block ratio threshold is what provides robustness.
    """
    top5 = (h >> 11) & 0x1F
    top4 = (h >> 12) & 0xF

    # Shift (LSL/LSR/ASR), add/sub, mov/cmp/add/sub immed
    if top5 in (0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07):
        return True

    # Data-processing register (010000)
    if (h & 0xFC00) == 0x4000:
        return True

    # Special data / branch exchange (010001)
    if (h & 0xFC00) == 0x4400:
        return True

    # LDR literal (01001)
    if (h & 0xF800) == 0x4800:
        return True

    # Load/store register offset (0101)
    if top4 == 0x5:
        return True

    # Load/store immediate offset (011/100)
    if top4 in (0x6, 0x7, 0x8):
        return True

    # Load/store halfword (1000)
    if top4 == 0x8:
        return True

    # SP-relative load/store (1001)
    if top4 == 0x9:
        return True

    # Load address (1010)
    if top4 == 0xA:
        return True

    # Add offset to SP / push/pop / stm/ldm / misc (1011)
    if top4 == 0xB:
        # PUSH/POP have recognizable patterns
        if (h & 0xFE00) in (0xB400, 0xBC00):
            return True
        # LDM/STM (1100 0xxx / 1100 1xxx in Thumb16 are 0xC000..0xCFFF, but some assemblers)
        return True

    # Conditional branch (1101)
    if top4 == 0xD:
        # 0xDE?? is UDF / svc-ish; still "valid-ish" for our purposes
        return True

    # Unconditional branch (11100)
    if (h & 0xF800) == 0xE000:
        return True

    return False


def is_thumb32_prefix(h1: int) -> bool:
    """
    Thumb-2 32-bit instructions have specific prefix ranges.
    We conservatively treat these as "potential 32-bit" if they look like it.
    Common prefixes: 11101, 11110, 11111 => 0xE800..0xFFFF (but not all are 32-bit)
    In practice, checking h1 top bits is good enough for block-level scoring.
    """
    return (h1 & 0xF800) in (0xE800, 0xF000, 0xF800)


def is_thumb32_pair(h1: int, h2: int) -> bool:
    """
    Conservative check: accept common 32-bit Thumb-2 patterns, reject obvious nonsense.
    We're not fully decoding — just avoiding counting totally random pairs as valid too often.
    """
    if not is_thumb32_prefix(h1):
        return False

    # Many 32-bit encodings have h2 with high bits 0b10xx or 0b11xx depending.
    # We'll accept broadly but reject the most unlikely: all-zero or all-ones.
    if h2 in (0x0000, 0xFFFF):
        return False

    # Accept common continuation patterns:
    # - For many Thumb-2 encodings, h2 has top bits 0b10xx xxxx xxxx xxxx (0x8000..0xBFFF)
    # - Or 0b11xx ... (0xC000..0xFFFF)
    if (h2 & 0x8000) == 0x8000:
        return True

    return True


def thumb_validity_mask(data: bytes) -> List[bool]:
    """
    Return a boolean list per halfword index indicating whether that halfword
    begins a valid Thumb instruction (16-bit) or a valid 32-bit instruction (marks first halfword only).
    """
    n = len(data) // 2
    ok = [False] * n
    i = 0
    while i < n:
        h1 = u16_le(data, 2 * i)

        # Try 32-bit first (Thumb-2)
        if i + 1 < n and is_thumb32_prefix(h1):
            h2 = u16_le(data, 2 * (i + 1))
            if is_thumb32_pair(h1, h2):
                ok[i] = True
                i += 2
                continue

        # Otherwise 16-bit
        if is_thumb16(h1):
            ok[i] = True

        i += 1

    return ok


def iter_kept_blocks(
    tail: bytes, ok_hw: List[bool], block_bytes: int, min_ratio: float
) -> Tuple[List[Tuple[int, int]], Dict]:
    """Return kept (start,end) byte ranges for blocks that pass min_ratio."""
    total_bytes = len(tail)
    if total_bytes < 2:
        return [], {
            "kept_blocks": 0,
            "total_blocks": 0,
            "kept_bytes": 0,
            "total_bytes": total_bytes,
            "kept_fraction": 0.0,
        }

    # Ensure block_bytes is even (halfword aligned)
    if block_bytes % 2 == 1:
        block_bytes -= 1
    if block_bytes < 2:
        block_bytes = 2

    blocks: List[Tuple[int, int]] = []
    total_blocks = 0
    kept_blocks = 0
    kept_bytes = 0

    for start in range(0, total_bytes, block_bytes):
        end = min(start + block_bytes, total_bytes)
        # ensure even end for halfword slicing
        end -= (end - start) % 2
        if end <= start:
            continue

        total_blocks += 1
        hw0 = start // 2
        hw1 = end // 2
        ok_slice = ok_hw[hw0:hw1]
        if not ok_slice:
            continue

        ratio = sum(1 for v in ok_slice if v) / len(ok_slice)
        if ratio >= min_ratio:
            blocks.append((start, end))
            kept_blocks += 1
            kept_bytes += (end - start)

    meta = {
        "kept_blocks": kept_blocks,
        "total_blocks": total_blocks,
        "kept_bytes": kept_bytes,
        "total_bytes": total_bytes,
        "kept_fraction": (kept_bytes / total_bytes) if total_bytes else 0.0,
    }
    return blocks, meta


def count_return_signatures(
    tail: bytes,
    ok_hw: List[bool],
    blocks: List[Tuple[int, int]],
    count_only_ok_starts: bool = False,
) -> Tuple[Dict[str, int], int]:
    """
    Count signature occurrences within kept blocks.
    Returns (counts, total_halfwords_counted).

    If count_only_ok_starts=True, only count on halfwords marked True in ok_hw
    (i.e., instruction starts per thumb_validity_mask). Otherwise count on all halfwords
    inside kept blocks (recommended; more stable).
    """
    counts = defaultdict(int)
    total_hw = 0

    for start, end in blocks:
        hw0 = start // 2
        hw1 = end // 2
        for i in range(hw0, hw1):
            if count_only_ok_starts and not ok_hw[i]:
                continue
            h = u16_le(tail, 2 * i)
            total_hw += 1

            # POP {...,PC} : 0xBDxx
            if (h & 0xFF00) == 0xBD00:
                counts["pop_pc"] += 1
            # BX LR : 0x4770
            if h == 0x4770:
                counts["bx_lr"] += 1
            # MOV PC, LR (common Thumb16 encoding): 0x46F7
            if h == 0x46F7:
                counts["mov_pc_lr"] += 1
            # PUSH {...,LR} : 0xB5xx
            if (h & 0xFF00) == 0xB500:
                counts["push_lr"] += 1

    return counts, total_hw


def analyze_file(
    path: Path, offset: int, block_bytes: int, min_ratio: float, count_only_ok: bool
) -> Tuple[Dict[str, int], Dict, int]:
    raw = path.read_bytes()
    if offset >= len(raw):
        return {}, {
            "kept_bytes": 0,
            "total_bytes": 0,
            "kept_fraction": 0.0,
            "kept_blocks": 0,
            "total_blocks": 0,
        }, 0

    tail = raw[offset:]
    # Ensure even length for halfword parsing; drop last byte if odd.
    if len(tail) % 2 == 1:
        tail = tail[:-1]

    ok_hw = thumb_validity_mask(tail)
    blocks, meta = iter_kept_blocks(tail, ok_hw, block_bytes, min_ratio)
    counts, total_hw = count_return_signatures(
        tail, ok_hw, blocks, count_only_ok_starts=count_only_ok
    )

    return counts, meta, total_hw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bin-root",
        required=True,
        help="Root folder containing per-device subfolders of firmware blobs",
    )
    ap.add_argument(
        "--pattern",
        default="*.dat",
        help="Glob pattern for firmware files within each device folder",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=6001,
        help="Byte offset to skip header before scanning",
    )
    ap.add_argument(
        "--block-bytes",
        type=int,
        default=2048,
        help="Block size in bytes for ratio test",
    )
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=0.9,
        help="Minimum ratio of valid instruction starts per block",
    )
    ap.add_argument("--out", default="out_code", help="Output directory")
    ap.add_argument(
        "--count-only-ok",
        action="store_true",
        help="Count signatures only on halfwords marked as instruction starts by thumb_validity_mask()",
    )
    args = ap.parse_args()

    bin_root = Path(args.bin_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dev_dir in sorted([p for p in bin_root.iterdir() if p.is_dir()]):
        device = dev_dir.name

        agg_counts = defaultdict(int)
        agg_total_hw = 0
        agg_kept_bytes = 0
        agg_total_bytes = 0

        used_files = 0
        for fw in dev_dir.glob(args.pattern):
            counts, meta, total_hw = analyze_file(
                fw, args.offset, args.block_bytes, args.min_ratio, args.count_only_ok
            )

            agg_kept_bytes += int(meta.get("kept_bytes", 0))
            agg_total_bytes += int(meta.get("total_bytes", 0))
            agg_total_hw += int(total_hw)
            for k, v in counts.items():
                agg_counts[k] += int(v)
            used_files += 1

        if used_files == 0:
            continue

        denom = agg_total_hw if agg_total_hw > 0 else 1
        kept_frac = (agg_kept_bytes / agg_total_bytes) if agg_total_bytes else 0.0

        rows.append(
            {
                "device": device,
                "kept_frac": kept_frac,
                "total_halfwords_counted": agg_total_hw,
                "pop_pc_per_1k": 1000.0 * agg_counts["pop_pc"] / denom,
                "bx_lr_per_1k": 1000.0 * agg_counts["bx_lr"] / denom,
                "mov_pc_lr_per_1k": 1000.0 * agg_counts["mov_pc_lr"] / denom,
                "push_lr_per_1k": 1000.0 * agg_counts["push_lr"] / denom,
            }
        )

    if not rows:
        raise SystemExit(
            "No device folders / firmware files matched. Check --bin-root and --pattern."
        )

    out_csv = out_dir / f"thumb_return_signatures_offset{args.offset}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()