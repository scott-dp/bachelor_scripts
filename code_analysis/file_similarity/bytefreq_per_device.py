#!/usr/bin/env python3
"""
bytefreq_per_device.py

Compute per-device byte frequency distributions (bytes 0..255) using all .dat files
for each device, starting after a header offset (default 6000 bytes).
Also computes similarity matrices between devices:
  - Jensen-Shannon distance (JSD)
  - Chi-square distance
  - Cosine similarity of SORTED histograms (helps detect monoalphabetic byte substitution)

Usage:
  python bytefreq_per_device.py --bin-root ../bin --offset 6000 --out out_freq
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict

def read_tail_bytes(path: Path, offset: int) -> bytes:
    data = path.read_bytes()
    if offset >= len(data):
        return b""
    return data[offset:]

def normalize_hist(counts: list[int]) -> list[float]:
    s = sum(counts)
    if s == 0:
        return [0.0]*256
    return [c/s for c in counts]

def js_distance(p: list[float], q: list[float]) -> float:
    # Jensen–Shannon distance = sqrt(JS divergence)
    # JS(P,Q)=0.5*KL(P||M)+0.5*KL(Q||M), M=(P+Q)/2
    def kl(a, b):
        out = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                out += ai * math.log2(ai/bi)
        return out
    m = [(pi+qi)/2 for pi, qi in zip(p, q)]
    js = 0.5*kl(p, m) + 0.5*kl(q, m)
    return math.sqrt(max(js, 0.0))

def chi2_distance(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    # symmetric chi-square distance
    s = 0.0
    for pi, qi in zip(p, q):
        denom = pi + qi + eps
        s += (pi - qi) * (pi - qi) / denom
    return 0.5 * s

def cosine_similarity(a: list[float], b: list[float], eps: float = 1e-12) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + eps)

def write_matrix_csv(path: Path, devices: list[str], mat: dict[tuple[str,str], float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["device"] + devices)
        for d1 in devices:
            row = [d1]
            for d2 in devices:
                row.append(f"{mat[(d1,d2)]:.6f}")
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-root", required=True, help="Root dir containing device folders with .dat files")
    ap.add_argument("--offset", type=int, default=6000, help="Start offset (header length), default 6000")
    ap.add_argument("--out", default="out_freq", help="Output directory")
    ap.add_argument("--pattern", default="*.dat", help="Glob pattern for firmware files (default *.dat)")
    args = ap.parse_args()

    bin_root = Path(args.bin_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device -> counts[256]
    dev_counts: dict[str, list[int]] = {}
    dev_files: dict[str, int] = {}

    # If layout is bin_root/<device>/*.dat
    for dev_dir in sorted([p for p in bin_root.iterdir() if p.is_dir()]):
        device = dev_dir.name
        counts = [0]*256
        nfiles = 0
        for fw in dev_dir.glob(args.pattern):
            tail = read_tail_bytes(fw, args.offset)
            if not tail:
                continue
            nfiles += 1
            for b in tail:
                counts[b] += 1
        if nfiles > 0:
            dev_counts[device] = counts
            dev_files[device] = nfiles

    devices = sorted(dev_counts.keys())
    if not devices:
        raise SystemExit("No device folders with matching firmware files found.")

    # normalize
    dev_probs = {d: normalize_hist(dev_counts[d]) for d in devices}

    # write per-device histogram CSV
    hist_csv = out_dir / "bytefreq_by_device.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["device", "num_files"] + [f"b{v:02x}" for v in range(256)])
        for d in devices:
            w.writerow([d, dev_files[d]] + [f"{p:.10f}" for p in dev_probs[d]])

    # similarity matrices
    jsd = {}
    chi2 = {}
    sorted_cos = {}
    for d1 in devices:
        for d2 in devices:
            p = dev_probs[d1]
            q = dev_probs[d2]
            jsd[(d1,d2)] = js_distance(p, q)
            chi2[(d1,d2)] = chi2_distance(p, q)
            # Sorted histogram comparison (perm-invariant-ish for substitution-cipher suspicion)
            sp = sorted(p)
            sq = sorted(q)
            sorted_cos[(d1,d2)] = cosine_similarity(sp, sq)

    write_matrix_csv(out_dir / "similarity_jsd.csv", devices, jsd)
    write_matrix_csv(out_dir / "similarity_chi2.csv", devices, chi2)
    write_matrix_csv(out_dir / "similarity_sorted_cosine.csv", devices, sorted_cos)

    # quick console summary: closest pairs by JSD, and closest by sorted-cosine
    pairs = []
    pairs2 = []
    for i in range(len(devices)):
        for j in range(i+1, len(devices)):
            d1, d2 = devices[i], devices[j]
            pairs.append((jsd[(d1,d2)], d1, d2))
            pairs2.append((sorted_cos[(d1,d2)], d1, d2))
    pairs.sort()
    pairs2.sort(reverse=True)

    print(f"Wrote outputs to: {out_dir.resolve()}")
    print("\nClosest device pairs by Jensen–Shannon distance (smaller = more similar):")
    for dist, d1, d2 in pairs[:10]:
        print(f"  {dist:.6f}  {d1}  vs  {d2}")

    print("\nMost similar pairs by SORTED-histogram cosine (closer to 1 = more 'permutation-like'):")
    for sim, d1, d2 in pairs2[:10]:
        print(f"  {sim:.6f}  {d1}  vs  {d2}")

    print("\nInterpretation tips:")
    print("  - If JSD is low AND normal cosine is high, they likely share similar byte semantics.")
    print("  - If JSD is high BUT sorted-cosine ~1.0, that’s consistent with a byte substitution/permutation.")
    print("  - If all devices look ~uniform (high entropy), suspect compression/encryption instead of opcodes.")

if __name__ == "__main__":
    main()
