"""
Mine common prologue/epilogue byte patterns around candidate subroutine boundaries
from a CSV diff report.

Usage:
  python subroutine_boundaries_csv.py --report diff.csv --firmware-dir /path/to/fw \
      --window 64 --min-len 4 --max-len 16 --top 30 --out report_out

CSV assumptions:
- The CSV contains columns like:
    base_file,new_file,side,start,end,note
- The side column indicates which firmware file contains the bytes to examine:
    * base -> base_file
    * new  -> new_file
- If side is missing/blank, insertion defaults to new_file, deletion defaults to
  base_file, otherwise new_file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Candidate:
    start: int
    end: int
    note: str
    base_file: str
    new_file: str
    side_file: str


def clamp(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))


def bytes_to_hex(b: bytes) -> str:
    return " ".join(f"{x:02x}" for x in b)


def pick_side_file(row: dict, base_file: str, new_file: str, note: str) -> str:
    """
    Decide which firmware file contains the region described by this CSV row.
    Prefer the explicit `side` column when present.
    """
    side = (row.get("side") or "").strip().lower()
    if side == "base":
        return base_file
    if side == "new":
        return new_file

    note_lower = note.lower()
    if "insertion" in note_lower:
        return new_file
    if "deletion" in note_lower:
        return base_file
    return new_file


def parse_candidates_from_csv(path: Path) -> List[Candidate]:
    out: List[Candidate] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"base_file", "new_file", "start", "end", "note"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for i, row in enumerate(reader, start=2):
            base_file = (row.get("base_file") or "").strip()
            new_file = (row.get("new_file") or "").strip()
            note = (row.get("note") or "").strip()
            start_raw = (row.get("start") or "").strip()
            end_raw = (row.get("end") or "").strip()

            if not base_file or not new_file:
                raise ValueError(f"Row {i}: missing base_file or new_file")
            if not start_raw or not end_raw:
                raise ValueError(f"Row {i}: missing start or end")

            start = int(start_raw)
            end = int(end_raw)
            side_file = pick_side_file(row, base_file, new_file, note)

            out.append(
                Candidate(
                    start=start,
                    end=end,
                    note=note,
                    base_file=base_file,
                    new_file=new_file,
                    side_file=side_file,
                )
            )

    return out


def load_firmware_bytes(fw_path: Path) -> bytes:
    return fw_path.read_bytes()


def collect_boundary_windows(
    fw: bytes, start: int, end: int, window: int
) -> Dict[str, bytes]:
    """
    Returns four windows:
      - start_pre: bytes [start-window, start)
      - start_post: bytes [start, start+window)
      - end_pre: bytes [end-window, end)
      - end_post: bytes [end, end+window)
    """
    n = len(fw)

    s0 = clamp(start - window, 0, n)
    s1 = clamp(start, 0, n)
    s2 = clamp(start + window, 0, n)

    e0 = clamp(end - window, 0, n)
    e1 = clamp(end, 0, n)
    e2 = clamp(end + window, 0, n)

    return {
        "start_pre": fw[s0:s1],
        "start_post": fw[s1:s2],
        "end_pre": fw[e0:e1],
        "end_post": fw[e1:e2],
    }


def mine_prefixes(windows: List[bytes], min_len: int, max_len: int) -> Counter:
    """
    Frequent prefixes: for each window, count every prefix length L.
    """
    c = Counter()
    for w in windows:
        for L in range(min_len, min(max_len, len(w)) + 1):
            c[w[:L]] += 1
    return c


def mine_suffixes(windows: List[bytes], min_len: int, max_len: int) -> Counter:
    """
    Frequent suffixes: for each window, count every suffix length L.
    """
    c = Counter()
    for w in windows:
        for L in range(min_len, min(max_len, len(w)) + 1):
            c[w[-L:]] += 1
    return c


def mine_kgrams(windows: List[bytes], k: int) -> Counter:
    c = Counter()
    for w in windows:
        if len(w) < k:
            continue
        for i in range(0, len(w) - k + 1):
            c[w[i:i + k]] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to CSV report")
    ap.add_argument("--firmware-dir", required=True, help="Directory containing .dat firmware files")
    ap.add_argument("--window", type=int, default=64, help="Bytes around boundary to analyze")
    ap.add_argument("--min-len", type=int, default=4, help="Min byte-length for prefix/suffix patterns")
    ap.add_argument("--max-len", type=int, default=16, help="Max byte-length for prefix/suffix patterns")
    ap.add_argument("--top", type=int, default=30, help="Top-N patterns to print")
    ap.add_argument("--out", default="boundary_mining_out", help="Output dir")
    ap.add_argument("--kgrams", default="4,6,8,10,12", help="Comma-separated k sizes for k-gram frequency")
    args = ap.parse_args()

    report_path = Path(args.report)
    fw_dir = Path(args.firmware_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = parse_candidates_from_csv(report_path)

    if not candidates:
        print("No candidates found in CSV. Check formatting.")
        return 2

    needed_files = sorted({c.side_file for c in candidates})
    firmwares: Dict[str, bytes] = {}
    missing = []

    # Index firmware files recursively so --firmware-dir can point at a root like bin/
    recursive_index: Dict[str, Path] = {}
    for p in fw_dir.rglob('*.dat'):
        recursive_index.setdefault(p.name, p)

    for fn in needed_files:
        direct = fw_dir / fn
        resolved: Path | None = None

        if direct.exists():
            resolved = direct
        elif fn in recursive_index:
            resolved = recursive_index[fn]

        if resolved is None:
            missing.append(fn)
        else:
            firmwares[fn] = load_firmware_bytes(resolved)

    if missing:
        print("Missing firmware files in --firmware-dir:")
        for m in missing:
            print(f"  - {m}")
        return 3

    start_post_windows: List[bytes] = []
    end_pre_windows: List[bytes] = []
    start_pre_windows: List[bytes] = []
    end_post_windows: List[bytes] = []

    per_candidate_dump = []

    for c in candidates:
        fw = firmwares[c.side_file]
        w = collect_boundary_windows(fw, c.start, c.end, args.window)

        start_pre_windows.append(w["start_pre"])
        start_post_windows.append(w["start_post"])
        end_pre_windows.append(w["end_pre"])
        end_post_windows.append(w["end_post"])

        per_candidate_dump.append({
            "base_file": c.base_file,
            "new_file": c.new_file,
            "side_file": c.side_file,
            "start": c.start,
            "end": c.end,
            "note": c.note,
            "windows_hex": {k: bytes_to_hex(v) for k, v in w.items()},
        })

    pro_prefix = mine_prefixes(start_post_windows, args.min_len, args.max_len)
    pro_suffix = mine_suffixes(start_pre_windows, args.min_len, args.max_len)

    epi_suffix = mine_suffixes(end_pre_windows, args.min_len, args.max_len)
    epi_prefix = mine_prefixes(end_post_windows, args.min_len, args.max_len)

    k_list = [int(x.strip()) for x in args.kgrams.split(",") if x.strip()]
    kgram_reports = {}
    for k in k_list:
        c1 = mine_kgrams(start_post_windows, k)
        c2 = mine_kgrams(end_pre_windows, k)
        kgram_reports[k] = {
            "start_post_top": [(bytes_to_hex(b), n) for b, n in c1.most_common(args.top)],
            "end_pre_top": [(bytes_to_hex(b), n) for b, n in c2.most_common(args.top)],
        }

    def top_patterns(counter: Counter) -> List[Tuple[str, int]]:
        items = sorted(counter.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
        return [(bytes_to_hex(b), n) for b, n in items[:args.top]]

    result = {
        "meta": {
            "report": str(report_path),
            "firmware_dir": str(fw_dir),
            "window": args.window,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "top": args.top,
            "num_candidates": len(candidates),
        },
        "top_patterns": {
            "prologue_prefix_start_post": top_patterns(pro_prefix),
            "prologue_suffix_start_pre": top_patterns(pro_suffix),
            "epilogue_suffix_end_pre": top_patterns(epi_suffix),
            "epilogue_prefix_end_post": top_patterns(epi_prefix),
        },
        "kgrams": kgram_reports,
        "candidates": per_candidate_dump,
    }

    json_path = out_dir / "boundary_patterns.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Parsed candidates: {len(candidates)}")
    print(f"Wrote: {json_path}")
    print()

    print("=== Top prologue-like prefixes (start_post) ===")
    for hx, n in result["top_patterns"]["prologue_prefix_start_post"]:
        print(f"{n:>4}  {hx}")

    print("\n=== Top epilogue-like suffixes (end_pre) ===")
    for hx, n in result["top_patterns"]["epilogue_suffix_end_pre"]:
        print(f"{n:>4}  {hx}")

    print("\n=== Top k-grams (start_post / end_pre) ===")
    for k in k_list:
        print(f"\n-- k={k} start_post --")
        for hx, n in result["kgrams"][k]["start_post_top"][:10]:
            print(f"{n:>4}  {hx}")
        print(f"-- k={k} end_pre --")
        for hx, n in result["kgrams"][k]["end_pre_top"][:10]:
            print(f"{n:>4}  {hx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
