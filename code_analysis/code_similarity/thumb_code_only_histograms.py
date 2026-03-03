#!/usr/bin/env python3
"""
thumb_code_only_histograms.py

Step 1: "code-only" byte histograms per device after Thumb filtering.

Heuristic Thumb/Thumb-2 validity decoder:
- Decodes little-endian halfwords
- Recognizes broad 16-bit Thumb encodings
- Handles common 32-bit Thumb-2 prefixes conservatively
- Scores blocks by valid-instruction ratio
- Keeps only blocks above a threshold, then builds byte histograms from kept bytes

Outputs:
- out_dir/code_only_bytefreq_by_device.csv
- out_dir/code_only_similarity_jsd.csv
- out_dir/code_only_similarity_chi2.csv
- out_dir/code_only_similarity_cosine.csv
- out_dir/code_only_summary.json

Usage:
  python thumb_code_only_histograms.py --bin-root ../bin --offset 6000 --out out_code --block-bytes 4096 --min-ratio 0.80
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------
# Basic stats / distances
# -----------------------------

def normalize(counts: List[int]) -> List[float]:
    s = sum(counts)
    if s == 0:
        return [0.0] * 256
    return [c / s for c in counts]

def js_distance(p: List[float], q: List[float]) -> float:
    # Jensen–Shannon distance (sqrt of divergence), base-2 logs
    def kl(a: List[float], b: List[float]) -> float:
        out = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                out += ai * math.log2(ai / bi)
        return out

    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return math.sqrt(max(js, 0.0))

def chi2_distance(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        denom = pi + qi + eps
        s += (pi - qi) * (pi - qi) / denom
    return 0.5 * s

def cosine_similarity(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    dot = sum(a*b for a, b in zip(p, q))
    np = math.sqrt(sum(a*a for a in p))
    nq = math.sqrt(sum(b*b for b in q))
    return dot / (np*nq + eps)

def write_matrix_csv(path: Path, devices: List[str], mat: Dict[Tuple[str, str], float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["device"] + devices)
        for d1 in devices:
            row = [d1] + [f"{mat[(d1, d2)]:.6f}" for d2 in devices]
            w.writerow(row)

# -----------------------------
# Thumb validity heuristics
# -----------------------------

def u16_le(data: bytes, i: int) -> int:
    return data[i] | (data[i+1] << 8)

def is_thumb16(h: int) -> bool:
    """
    Heuristic recognizer for 16-bit Thumb (covers most common encodings).

    IMPORTANT: This is intentionally permissive (we want "likely code blocks",
    not perfect disassembly). The block ratio threshold is what provides robustness.
    """
    top5 = (h >> 11) & 0x1F
    top4 = (h >> 12) & 0xF

    # Shift (LSL/LSR/ASR), add/sub, mov/cmp/add/sub immediates
    if 0x00 <= top5 <= 0x03:  # 000xx shifts + add/sub reg
        return True
    if 0x04 <= top5 <= 0x07:  # 001xx add/sub/cmp/mov imm
        return True

    # Data-processing register, special, branch/exchange
    if (h & 0xFC00) == 0x4000:  # 010000 data-processing
        return True
    if (h & 0xFC00) == 0x4400:  # 010001 special data, BX/BLX
        return True

    # LDR literal
    if (h & 0xF800) == 0x4800:  # 01001 Rt, [PC, imm]
        return True

    # Load/store register offset, immediate offset, halfword
    if (h & 0xF000) == 0x5000:  # 0101xx load/store reg offset
        return True
    if (h & 0xE000) == 0x6000:  # 011xxx load/store imm offset (word/byte)
        return True
    if (h & 0xF000) == 0x8000:  # 1000x load/store halfword imm
        return True

    # Load/store SP-relative
    if (h & 0xF000) == 0x9000:  # 1001 Rt, [SP, imm]
        return True

    # ADD to PC/SP, stack adjust
    if (h & 0xF000) == 0xA000:  # 1010 add to PC/SP
        return True
    if (h & 0xFF00) == 0xB000:  # 1011 0000 add sp/sub sp (also misc)
        # Exclude some very rare/reserved patterns? Keep permissive.
        return True

    # PUSH/POP
    if (h & 0xFE00) == 0xB400:  # push
        return True
    if (h & 0xFE00) == 0xBC00:  # pop
        return True

    # Multiple load/store
    if (h & 0xF000) == 0xC000:  # STM/LDM
        return True

    # Conditional branch, SWI, unconditional branch
    if (h & 0xF000) == 0xD000:
        # 1101 cond branch / SWI; 0xDE?? is UDF in Thumb (often "undefined")
        # But UDF can appear in real code; still treat as valid-ish in code blocks.
        return True
    if (h & 0xF800) == 0xE000:  # unconditional B
        return True

    # If none matched, assume invalid
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
    # We'll accept a wide range but reject if h2 looks like "all zeros" or extreme.
    if h2 == 0x0000 or h2 == 0xFFFF:
        return False

    # If h1 is 0xF000/0xF800 class (often BL/branches), h2 often has high bit set.
    if (h1 & 0xF000) == 0xF000:
        return (h2 & 0x8000) != 0

    # If h1 is 0xE800 class (load/store multiple, etc.), accept broadly.
    if (h1 & 0xF800) == 0xE800:
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
        h1 = u16_le(data, 2*i)
        # Try 32-bit first (Thumb-2)
        if i + 1 < n and is_thumb32_prefix(h1):
            h2 = u16_le(data, 2*(i+1))
            if is_thumb32_pair(h1, h2):
                ok[i] = True
                # We skip the second halfword as part of this instruction.
                i += 2
                continue

        # Otherwise 16-bit
        if is_thumb16(h1):
            ok[i] = True
        i += 1
    return ok

# -----------------------------
# Code-only histogram per device
# -----------------------------

def analyze_file_code_bytes(path: Path, offset: int, block_bytes: int, min_ratio: float) -> Tuple[List[int], Dict]:
    """
    Returns (hist[256], meta) using only code-like blocks.
    """
    raw = path.read_bytes()
    if offset >= len(raw):
        return [0]*256, {"kept_blocks": 0, "total_blocks": 0, "kept_bytes": 0, "total_bytes": 0}

    tail = raw[offset:]
    # Ensure even length for halfword parsing; drop last byte if odd.
    if len(tail) % 2 == 1:
        tail = tail[:-1]

    ok_hw = thumb_validity_mask(tail)
    total_bytes = len(tail)

    # Block over bytes (must be multiple of 2 for halfword alignment)
    if block_bytes < 256:
        block_bytes = 256
    if block_bytes % 2 == 1:
        block_bytes += 1

    hist = [0]*256
    kept_blocks = 0
    total_blocks = 0
    kept_bytes = 0

    # block index in bytes
    for b0 in range(0, len(tail), block_bytes):
        b1 = min(len(tail), b0 + block_bytes)
        if b1 - b0 < 2:
            continue

        # Convert to halfword indices
        hw0 = b0 // 2
        hw1 = b1 // 2
        block_ok = ok_hw[hw0:hw1]
        if not block_ok:
            continue
        total_blocks += 1
        ratio = sum(1 for x in block_ok if x) / len(block_ok)

        if ratio >= min_ratio:
            kept_blocks += 1
            block = tail[b0:b1]
            kept_bytes += len(block)
            for byt in block:
                hist[byt] += 1

    meta = {
        "kept_blocks": kept_blocks,
        "total_blocks": total_blocks,
        "kept_bytes": kept_bytes,
        "total_bytes": total_bytes,
        "kept_fraction": (kept_bytes / total_bytes) if total_bytes else 0.0,
    }
    return hist, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-root", required=True, help="Root dir with per-device subfolders containing .dat files")
    ap.add_argument("--offset", type=int, default=6000, help="Start offset after header (default 6000)")
    ap.add_argument("--pattern", default="*.dat", help="Glob pattern for firmware files (default *.dat)")
    ap.add_argument("--out", default="out_code", help="Output directory")
    ap.add_argument("--block-bytes", type=int, default=4096, help="Block size for validity ratio (default 4096)")
    ap.add_argument("--min-ratio", type=float, default=0.80, help="Min valid-halfword ratio to keep block (default 0.80)")
    args = ap.parse_args()

    bin_root = Path(args.bin_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_hist: Dict[str, List[int]] = {}
    dev_meta: Dict[str, Dict] = {}

    for dev_dir in sorted([p for p in bin_root.iterdir() if p.is_dir()]):
        device = dev_dir.name
        agg = [0]*256
        file_metas = []
        nfiles = 0

        for fw in dev_dir.glob(args.pattern):
            h, m = analyze_file_code_bytes(fw, args.offset, args.block_bytes, args.min_ratio)
            if sum(h) == 0:
                continue
            nfiles += 1
            for i in range(256):
                agg[i] += h[i]
            m["file"] = fw.name
            file_metas.append(m)

        if nfiles > 0:
            dev_hist[device] = agg
            dev_meta[device] = {
                "num_files_used": nfiles,
                "offset": args.offset,
                "block_bytes": args.block_bytes,
                "min_ratio": args.min_ratio,
                "files": file_metas,
                "total_kept_bytes": sum(m["kept_bytes"] for m in file_metas),
                "total_bytes": sum(m["total_bytes"] for m in file_metas),
            }

    devices = sorted(dev_hist.keys())
    if not devices:
        raise SystemExit("No devices produced any kept code-like blocks. Try lowering --min-ratio to 0.70.")

    dev_prob = {d: normalize(dev_hist[d]) for d in devices}

    # Write histograms
    hist_csv = out_dir / "code_only_bytefreq_by_device.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["device"] + [f"b{v:02x}" for v in range(256)])
        for d in devices:
            w.writerow([d] + [f"{p:.10f}" for p in dev_prob[d]])

    # Similarity matrices
    jsd = {}
    chi2 = {}
    cos = {}
    for d1 in devices:
        for d2 in devices:
            p, q = dev_prob[d1], dev_prob[d2]
            jsd[(d1, d2)] = js_distance(p, q)
            chi2[(d1, d2)] = chi2_distance(p, q)
            cos[(d1, d2)] = cosine_similarity(p, q)

    write_matrix_csv(out_dir / "code_only_similarity_jsd.csv", devices, jsd)
    write_matrix_csv(out_dir / "code_only_similarity_chi2.csv", devices, chi2)
    write_matrix_csv(out_dir / "code_only_similarity_cosine.csv", devices, cos)

    # Summary JSON
    summary = {
        "devices": devices,
        "meta": dev_meta,
        "notes": {
            "meaning": "Histograms computed only from blocks with high Thumb validity ratio",
            "tip": "If no blocks are kept for a device, lower --min-ratio or increase --block-bytes",
        },
    }
    (out_dir / "code_only_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console: show kept fractions
    print(f"Wrote outputs to: {out_dir.resolve()}\n")
    print("Per-device kept fraction (code-like bytes / total bytes after header):")
    for d in devices:
        tm = dev_meta[d]
        total = tm["total_bytes"] or 1
        kept = tm["total_kept_bytes"]
        frac = kept / total
        print(f"  {d:10s}  kept={kept:10d}  total={total:10d}  frac={frac:.3f}")

    # Closest pairs by JSD
    pairs = []
    for i in range(len(devices)):
        for j in range(i+1, len(devices)):
            d1, d2 = devices[i], devices[j]
            pairs.append((jsd[(d1, d2)], d1, d2))
    pairs.sort()
    print("\nClosest device pairs by CODE-ONLY Jensen–Shannon distance:")
    for dist, d1, d2 in pairs[:10]:
        print(f"  {dist:.6f}  {d1} vs {d2}")

if __name__ == "__main__":
    main()
