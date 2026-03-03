#!/usr/bin/env python3
"""
thumb_instruction_class_frequencies.py

Step 2: Instruction-class frequency analysis on STRICT code-like blocks.

Uses:
- same Thumb validity heuristics as Step 1
- block-level filtering (block-bytes + min-ratio)
- counts instruction classes per kept block
- normalizes per device

Usage:
  python thumb_instruction_class_frequencies.py \
    --bin-root ../bin \
    --offset 6000 \
    --block-bytes 2048 \
    --min-ratio 0.90 \
    --out out_instr
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

# ---------------- Thumb helpers (same as before) ----------------

def u16_le(data: bytes, i: int) -> int:
    return data[i] | (data[i+1] << 8)

def is_thumb16(h: int) -> bool:
    if 0x0000 <= (h >> 11) <= 0x03: return True
    if 0x04 <= (h >> 11) <= 0x07: return True
    if (h & 0xFC00) in (0x4000, 0x4400): return True
    if (h & 0xF800) == 0x4800: return True
    if (h & 0xF000) == 0x5000: return True
    if (h & 0xE000) == 0x6000: return True
    if (h & 0xF000) == 0x8000: return True
    if (h & 0xF000) == 0x9000: return True
    if (h & 0xF000) == 0xA000: return True
    if (h & 0xFF00) == 0xB000: return True
    if (h & 0xFE00) in (0xB400, 0xBC00): return True
    if (h & 0xF000) == 0xC000: return True
    if (h & 0xF000) == 0xD000: return True
    if (h & 0xF800) == 0xE000: return True
    return False

def is_thumb32_prefix(h: int) -> bool:
    return (h & 0xF800) in (0xE800, 0xF000, 0xF800)

def thumb_valid_mask(data: bytes) -> List[bool]:
    n = len(data) // 2
    ok = [False] * n
    i = 0
    while i < n:
        h1 = u16_le(data, 2*i)
        if i+1 < n and is_thumb32_prefix(h1):
            ok[i] = True
            i += 2
            continue
        if is_thumb16(h1):
            ok[i] = True
        i += 1
    return ok

# ---------------- Instruction classification ----------------

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

    # Arithmetic / data processing
    if (h & 0xFC00) == 0x4000: return "alu"
    if 0x0000 <= (h >> 11) <= 0x07: return "alu"

    return None

# ---------------- Main analysis ----------------

def analyze_device(path: Path, offset: int, block_bytes: int, min_ratio: float) -> Counter:
    raw = path.read_bytes()
    if offset >= len(raw):
        return Counter()

    data = raw[offset:]
    if len(data) % 2:
        data = data[:-1]

    ok = thumb_valid_mask(data)
    counts = Counter()

    for b0 in range(0, len(data), block_bytes):
        b1 = min(len(data), b0 + block_bytes)
        hw0, hw1 = b0//2, b1//2
        block_ok = ok[hw0:hw1]
        if not block_ok:
            continue
        if sum(block_ok) / len(block_ok) < min_ratio:
            continue

        for i in range(hw0, hw1):
            if not ok[i]:
                continue
            h = u16_le(data, 2*i)
            cls = classify_halfword(h)
            if cls:
                counts[cls] += 1

    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-root", required=True)
    ap.add_argument("--offset", type=int, default=6000)
    ap.add_argument("--block-bytes", type=int, default=2048)
    ap.add_argument("--min-ratio", type=float, default=0.90)
    ap.add_argument("--out", default="out_instr")
    ap.add_argument("--pattern", default="*.dat")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Counter] = {}

    for dev_dir in sorted(p for p in Path(args.bin_root).iterdir() if p.is_dir()):
        agg = Counter()
        nfiles = 0
        for fw in dev_dir.glob(args.pattern):
            c = analyze_device(fw, args.offset, args.block_bytes, args.min_ratio)
            if c:
                agg.update(c)
                nfiles += 1
        if nfiles:
            results[dev_dir.name] = agg

    # Normalize + write CSV
    classes = sorted({k for c in results.values() for k in c})
    csv_path = out_dir / "instruction_class_frequencies.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["device"] + classes)
        for dev, cnt in results.items():
            total = sum(cnt.values()) or 1
            w.writerow([dev] + [f"{cnt.get(c,0)/total:.6f}" for c in classes])

    # Raw counts JSON
    (out_dir / "instruction_class_counts.json").write_text(
        json.dumps({d: dict(c) for d,c in results.items()}, indent=2),
        encoding="utf-8"
    )

    print(f"Wrote instruction-class frequencies to {csv_path.resolve()}")

if __name__ == "__main__":
    main()
