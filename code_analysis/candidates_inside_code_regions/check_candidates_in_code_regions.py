#!/usr/bin/env python3
"""
check_candidates_in_code_regions.py

For each candidate region in filtered.csv:
- pick the relevant firmware blob (base_file vs new_file based on side)
- compute "kept code-like blocks" using the SAME permissive Thumb heuristic and block ratio test
  from thumb_return_signatures.py
- check whether candidate [start,end) is contained in / overlaps any kept block.

Outputs:
- candidate_code_region_hits.csv (per-row flags)
- candidate_code_region_summary.csv (per-device + overall summary)

Example:
  python check_candidates_in_code_regions.py \
    --csv /mnt/data/filtered.csv \
    --bin-root /path/to/bin \
    --offset 6001 \
    --block-bytes 2048 \
    --min-ratio 0.9 \
    --out out_candidates
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import the exact permissive logic from your script:
# (Make sure thumb_return_signatures.py is in the same folder as this script,
#  or pass --trs-path to point to it.)
import importlib.util
import sys


@dataclass(frozen=True)
class KeptBlocksResult:
    blocks: List[Tuple[int, int]]  # list of (start,end) offsets in BYTES, relative to file start
    meta: Dict[str, Any]


def load_trs_module(trs_path: Path):
    spec = importlib.util.spec_from_file_location("thumb_return_signatures", str(trs_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {trs_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["thumb_return_signatures"] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_kept_blocks_for_file(
    trs_mod,
    fw_path: Path,
    offset: int,
    block_bytes: int,
    min_ratio: float,
) -> KeptBlocksResult:
    raw = fw_path.read_bytes()
    if offset >= len(raw):
        return KeptBlocksResult(blocks=[], meta={
            "kept_blocks": 0, "total_blocks": 0, "kept_bytes": 0,
            "total_bytes": 0, "kept_fraction": 0.0
        })

    tail = raw[offset:]
    if len(tail) % 2 == 1:
        tail = tail[:-1]

    ok_hw = trs_mod.thumb_validity_mask(tail)
    # iter_kept_blocks returns blocks relative to tail; shift them by offset to become file-relative
    blocks_tail, meta = trs_mod.iter_kept_blocks(tail, ok_hw, block_bytes, min_ratio)
    blocks_file = [(offset + s, offset + e) for (s, e) in blocks_tail]
    return KeptBlocksResult(blocks=blocks_file, meta=meta)


def is_contained(region: Tuple[int, int], block: Tuple[int, int]) -> bool:
    rs, re = region
    bs, be = block
    return rs >= bs and re <= be


def overlaps(region: Tuple[int, int], block: Tuple[int, int]) -> bool:
    rs, re = region
    bs, be = block
    return not (re <= bs or rs >= be)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to filtered.csv")
    ap.add_argument("--bin-root", required=True, help="Root folder containing per-device subfolders with firmware blobs")
    ap.add_argument("--trs-path", default="thumb_return_signatures.py", help="Path to thumb_return_signatures.py")
    ap.add_argument("--offset", type=int, default=6001, help="Byte offset to skip header before scanning for code blocks")
    ap.add_argument("--block-bytes", type=int, default=2048, help="Block size (bytes) for ratio test")
    ap.add_argument("--min-ratio", type=float, default=0.9, help="Minimum ratio of valid Thumb starts per block")
    ap.add_argument("--out", default="out_candidates", help="Output directory")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    bin_root = Path(args.bin_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trs_path = Path(args.trs_path)
    if not trs_path.exists():
        # allow relative-to-csv or cwd usage
        alt = (Path.cwd() / trs_path)
        if alt.exists():
            trs_path = alt
        else:
            raise SystemExit(f"thumb_return_signatures.py not found at: {trs_path}")

    trs_mod = load_trs_module(trs_path)

    # Cache kept blocks per firmware file path (string) to avoid re-scanning blobs repeatedly
    kept_cache: Dict[str, KeptBlocksResult] = {}

    rows_out: List[Dict[str, Any]] = []

    # Counters
    overall_total = 0
    overall_contained = 0
    overall_overlaps = 0
    per_device = {}  # device -> dict counters

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"device", "base_file", "new_file", "start", "end", "side"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

        for r in reader:
            device = r["device"].strip()
            side = r["side"].strip().lower()
            start = int(float(r["start"]))
            end = int(float(r["end"]))
            # Interpret candidate region as [start,end) (common convention).
            region = (start, end)

            fw_name = r["base_file"].strip() if side == "base" else r["new_file"].strip()
            fw_path = bin_root / device / fw_name

            hit_contained = False
            hit_overlaps = False
            kept_fraction = None
            kept_blocks_n = None

            if fw_path.exists():
                key = str(fw_path.resolve())
                if key not in kept_cache:
                    kept_cache[key] = compute_kept_blocks_for_file(
                        trs_mod, fw_path, args.offset, args.block_bytes, args.min_ratio
                    )
                kept = kept_cache[key]
                kept_fraction = kept.meta.get("kept_fraction", 0.0)
                kept_blocks_n = len(kept.blocks)

                # Check containment / overlap against any kept block
                for b in kept.blocks:
                    if not hit_overlaps and overlaps(region, b):
                        hit_overlaps = True
                    if not hit_contained and is_contained(region, b):
                        hit_contained = True
                    if hit_overlaps and hit_contained:
                        break

            # Update counters
            overall_total += 1
            overall_contained += int(hit_contained)
            overall_overlaps += int(hit_overlaps)

            if device not in per_device:
                per_device[device] = {"total": 0, "contained": 0, "overlaps": 0, "missing_fw": 0}
            per_device[device]["total"] += 1
            per_device[device]["contained"] += int(hit_contained)
            per_device[device]["overlaps"] += int(hit_overlaps)
            per_device[device]["missing_fw"] += int(not fw_path.exists())

            rows_out.append({
                **r,
                "fw_file_used": fw_name,
                "fw_exists": fw_path.exists(),
                "kept_blocks": kept_blocks_n,
                "kept_fraction": kept_fraction,
                "contained_in_code_region": hit_contained,
                "overlaps_code_region": hit_overlaps,
            })

    # Write row-level output
    out_hits = out_dir / "candidate_code_region_hits.csv"
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with out_hits.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    # Write summary output
    out_summary = out_dir / "candidate_code_region_summary.csv"
    summary_rows = []
    for dev, c in sorted(per_device.items()):
        total = c["total"] or 1
        summary_rows.append({
            "device": dev,
            "candidates_total": c["total"],
            "contained_hits": c["contained"],
            "contained_rate": c["contained"] / total,
            "overlap_hits": c["overlaps"],
            "overlap_rate": c["overlaps"] / total,
            "missing_firmware_rows": c["missing_fw"],
        })
    # Overall row
    total = overall_total or 1
    summary_rows.append({
        "device": "__OVERALL__",
        "candidates_total": overall_total,
        "contained_hits": overall_contained,
        "contained_rate": overall_contained / total,
        "overlap_hits": overall_overlaps,
        "overlap_rate": overall_overlaps / total,
        "missing_firmware_rows": sum(v["missing_fw"] for v in per_device.values()),
    })

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("Wrote:")
    print(" ", out_hits)
    print(" ", out_summary)


if __name__ == "__main__":
    main()
