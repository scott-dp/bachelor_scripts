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
  python thumb_code_only_histograms.py --bin-root ../bin --offset 6001 --out out_code --block-bytes 4096 --min-ratio 0.80
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
