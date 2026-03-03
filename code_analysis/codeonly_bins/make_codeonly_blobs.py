#!/usr/bin/env python3
"""
make_codeonly_blobs.py

Extract "plausible code" blocks using the SAME permissive Thumb block filter
as in thumb_return_signatures.py (validity mask + block ratio threshold),
then write concatenated "code-only" blobs for feeding to CpuRec.

Outputs (per input firmware file):
- <name>_codeonly.bin      : concatenated kept blocks
- <name>_codeonly_map.csv  : mapping from output offsets -> original offsets

Typical usage:
  python make_codeonly_blobs.py --bin-root ../../../bin --offset 6001 \
    --block-bytes 2048 --min-ratio 0.9 --out out_codeonly

You can also point it at individual files:
  python make_codeonly_blobs.py --inputs DUE6100-D.4.2.0.dat --offset 6001 \
    --block-bytes 2048 --min-ratio 0.9 --out out_codeonly
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def u16_le(data: bytes, i: int) -> int:
    return data[i] | (data[i + 1] << 8)


def is_thumb16(h: int) -> bool:
    # Same permissive recognizer as your script
    top5 = (h >> 11) & 0x1F
    top4 = (h >> 12) & 0xF

    if top5 in (0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07):
        return True
    if (h & 0xFC00) == 0x4000:
        return True
    if (h & 0xFC00) == 0x4400:
        return True
    if (h & 0xF800) == 0x4800:
        return True
    if top4 == 0x5:
        return True
    if top4 in (0x6, 0x7, 0x8):
        return True
    if top4 == 0x8:
        return True
    if top4 == 0x9:
        return True
    if top4 == 0xA:
        return True
    if top4 == 0xB:
        if (h & 0xFE00) in (0xB400, 0xBC00):
            return True
        return True
    if top4 == 0xD:
        return True
    if (h & 0xF800) == 0xE000:
        return True
    return False


def is_thumb32_prefix(h1: int) -> bool:
    return (h1 & 0xF800) in (0xE800, 0xF000, 0xF800)


def is_thumb32_pair(h1: int, h2: int) -> bool:
    if not is_thumb32_prefix(h1):
        return False
    if h2 in (0x0000, 0xFFFF):
        return False
    if (h2 & 0x8000) == 0x8000:
        return True
    return True


def thumb_validity_mask(data: bytes) -> List[bool]:
    """
    Boolean per halfword index: True if that halfword begins a valid Thumb16
    OR is the first halfword of an accepted Thumb-2 32-bit instruction.
    """
    n = len(data) // 2
    ok = [False] * n
    i = 0
    while i < n:
        h1 = u16_le(data, 2 * i)

        if i + 1 < n and is_thumb32_prefix(h1):
            h2 = u16_le(data, 2 * (i + 1))
            if is_thumb32_pair(h1, h2):
                ok[i] = True
                i += 2
                continue

        if is_thumb16(h1):
            ok[i] = True

        i += 1

    return ok


def iter_kept_blocks(
    tail: bytes, ok_hw: List[bool], block_bytes: int, min_ratio: float
) -> Tuple[List[Tuple[int, int]], Dict]:
    """Return kept (start,end) byte ranges (relative to tail) for blocks that pass min_ratio."""
    total_bytes = len(tail)
    if total_bytes < 2:
        return [], {
            "kept_blocks": 0,
            "total_blocks": 0,
            "kept_bytes": 0,
            "total_bytes": total_bytes,
            "kept_fraction": 0.0,
        }

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
        end -= (end - start) % 2  # halfword aligned
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


def process_one_file(
    fw_path: Path,
    out_dir: Path,
    offset: int,
    block_bytes: int,
    min_ratio: float,
    write_empty: bool = False,
) -> None:
    raw = fw_path.read_bytes()
    if offset >= len(raw):
        print(f"[skip] {fw_path} (offset beyond EOF)")
        return

    tail = raw[offset:]
    if len(tail) % 2 == 1:
        tail = tail[:-1]

    ok_hw = thumb_validity_mask(tail)
    blocks, meta = iter_kept_blocks(tail, ok_hw, block_bytes, min_ratio)

    stem = fw_path.stem
    out_bin = out_dir / f"{stem}_codeonly.bin"
    out_map = out_dir / f"{stem}_codeonly_map.csv"

    if not blocks and not write_empty:
        print(f"[no blocks] {fw_path} (kept_fraction={meta['kept_fraction']:.4f})")
        return

    # Concatenate kept blocks, and write mapping CSV
    out_bytes = bytearray()
    map_rows = []
    out_off = 0

    for (start, end) in blocks:
        chunk = tail[start:end]
        out_bytes.extend(chunk)

        # Map output offsets back to original file offsets (absolute)
        src_abs_start = offset + start
        src_abs_end = offset + end
        map_rows.append(
            {
                "out_start": out_off,
                "out_end": out_off + (end - start),
                "src_start": src_abs_start,
                "src_end": src_abs_end,
                "len": (end - start),
            }
        )
        out_off += (end - start)

    out_bin.write_bytes(bytes(out_bytes))
    with out_map.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["out_start", "out_end", "src_start", "src_end", "len"])
        w.writeheader()
        w.writerows(map_rows)

    print(
        f"[ok] {fw_path.name} -> {out_bin.name} "
        f"(kept_fraction={meta['kept_fraction']:.4f}, kept_blocks={meta['kept_blocks']}, bytes={len(out_bytes)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bin-root",
        help="Root folder containing per-device subfolders of firmware blobs (like your other scripts).",
    )
    group.add_argument(
        "--inputs",
        nargs="+",
        help="One or more firmware files to process directly.",
    )

    ap.add_argument("--pattern", default="*.dat", help="Glob within each device folder if using --bin-root")
    ap.add_argument("--offset", type=int, default=6001, help="Byte offset to skip header before scanning")
    ap.add_argument("--block-bytes", type=int, default=2048, help="Block size in bytes for ratio test")
    ap.add_argument("--min-ratio", type=float, default=0.9, help="Minimum valid-start ratio per block")
    ap.add_argument("--out", default="out_codeonly", help="Output directory")
    ap.add_argument(
        "--write-empty",
        action="store_true",
        help="Also write empty _codeonly.bin files if no blocks are kept (normally skipped).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fw_files: List[Path] = []

    if args.inputs:
        fw_files = [Path(p) for p in args.inputs]
    else:
        bin_root = Path(args.bin_root)
        for dev_dir in sorted([p for p in bin_root.iterdir() if p.is_dir()]):
            fw_files.extend(sorted(dev_dir.glob(args.pattern)))

    if not fw_files:
        raise SystemExit("No firmware files matched. Check --bin-root/--inputs and --pattern.")

    for fw in fw_files:
        if not fw.is_file():
            print(f"[skip] {fw} (not a file)")
            continue
        process_one_file(
            fw_path=fw,
            out_dir=out_dir,
            offset=args.offset,
            block_bytes=args.block_bytes,
            min_ratio=args.min_ratio,
            write_empty=args.write_empty,
        )


if __name__ == "__main__":
    main()