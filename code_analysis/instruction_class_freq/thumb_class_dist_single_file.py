#!/usr/bin/env python3
"""
thumb_class_dist_single_file.py

Read a single raw binary file that is assumed to be pure Thumb code,
classify halfwords into broad instruction classes, and print a distribution.

Usage:
  python thumb_class_dist_single_file.py path/to/firmware.bin
  python thumb_class_dist_single_file.py path/to/firmware.bin --offset 0x0 --count-thumb32
"""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

# ---------------- little-endian halfword ----------------

def u16_le(data: bytes, i: int) -> int:
    return data[i] | (data[i + 1] << 8)

# ---------------- Thumb-2 32-bit prefix detection ----------------
# Same heuristic you had: treat these as the first halfword of a 32-bit Thumb instruction.
def is_thumb32_prefix(h: int) -> bool:
    return (h & 0xF800) in (0xE800, 0xF000, 0xF800)

# ---------------- Instruction classification (16-bit only) ----------------

def classify_halfword(h: int) -> str | None:
    # Stack ops
    if (h & 0xFE00) == 0xB400: return "push"
    if (h & 0xFE00) == 0xBC00: return "pop"

    # BX / BLX
    if h == 0x4770: return "return"     # bx lr
    if (h & 0xFF87) == 0x4700: return "branch_reg"

    # Branches
    if (h & 0xF000) == 0xD000: return "branch_cond"
    if (h & 0xF800) == 0xE000: return "branch"

    # Load/store
    if (h & 0xF000) in (0x5000, 0x6000, 0x8000, 0x9000): return "load_store"

    # Multiple load/store
    if (h & 0xF000) == 0xC000: return "ldm_stm"

    # Arithmetic / data processing (very broad)
    if (h & 0xFC00) == 0x4000: return "alu"
    if 0x0000 <= (h >> 11) <= 0x07: return "alu"

    return None

# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--offset", type=lambda s: int(s, 0), default=0,
                    help="Start offset (decimal or 0x...)")
    ap.add_argument("--count-thumb32", action="store_true",
                    help="If set, detect Thumb-2 32-bit prefixes and count them as 'thumb32' (skips the next halfword).")
    args = ap.parse_args()

    raw = args.file.read_bytes()
    if args.offset >= len(raw):
        raise SystemExit(f"Offset {args.offset} >= file size {len(raw)}")

    data = raw[args.offset:]
    if len(data) < 2:
        raise SystemExit("Not enough data after offset to read any halfwords.")
    if len(data) % 2:
        data = data[:-1]  # drop last byte to keep halfword alignment

    counts = Counter()
    total_halfwords = len(data) // 2
    classified_halfwords = 0
    unknown_halfwords = 0

    i = 0
    while i < total_halfwords:
        h = u16_le(data, 2 * i)

        # Optional Thumb-2 32-bit handling:
        if args.count_thumb32 and is_thumb32_prefix(h) and (i + 1) < total_halfwords:
            counts["thumb32"] += 1
            classified_halfwords += 2  # you consumed 2 halfwords
            i += 2
            continue

        cls = classify_halfword(h)
        if cls is None:
            unknown_halfwords += 1
        else:
            counts[cls] += 1
            classified_halfwords += 1
        i += 1

    # Print results
    total_classified_items = sum(counts.values()) or 1  # avoid div by 0 in display
    print(f"File: {args.file}")
    print(f"Offset: {args.offset} bytes")
    print(f"Total halfwords scanned: {total_halfwords}")
    print(f"Classified halfwords (or thumb32 pairs if enabled): {classified_halfwords}")
    print(f"Unknown/unclassified halfwords: {unknown_halfwords}")
    print()

    print("Class distribution (count, fraction of classified):")
    for k, v in counts.most_common():
        print(f"  {k:12s}  {v:10d}  {v / total_classified_items:10.6f}")

if __name__ == "__main__":
    main()