"""
Mine common prologue/epilogue byte patterns around candidate subroutine boundaries
from a .docx diff report.

Usage:
  python mine_boundaries.py --report Document.docx --firmware-dir /path/to/fw \
      --window 64 --min-len 4 --max-len 16 --top 30 --out report_out

Assumptions (you can tweak):
- Lines like "Byte 5094 -> 5344 deletion from 2.8.2 ..." mean the region exists in firmware version 2.8.2
- Lines like "Byte 6301 -> 6352 insertion into 2.8.4 ..." mean the region exists in firmware version 2.8.4
- Firmware files are in firmware-dir and named exactly like in the arrow headers, e.g. "DUE6001-D.2.8.2.dat"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docx import Document


ARROW_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*->\s*([A-Za-z0-9_.-]+)\s*$")
CAND_RE = re.compile(
    r"Byte\s+(\d+)\s*->\s*(\d+)\s+(.+)$",
    re.IGNORECASE
)

# tries to catch "... insertion into 2.8.7 ..." or "... deletion from 2.8.4 ..."
INTO_VER_RE = re.compile(r"\binto\s+([0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
FROM_VER_RE = re.compile(r"\bfrom\s+([0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)


@dataclass
class Candidate:
    start: int
    end: int
    note: str
    base_file: str          # left side of A->B header
    new_file: str           # right side of A->B header
    side_file: str          # which file actually contains the bytes to examine
    # boundary windows will be derived later


def clamp(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))


def bytes_to_hex(b: bytes) -> str:
    return " ".join(f"{x:02x}" for x in b)


def read_docx_lines(path: Path) -> List[str]:
    doc = Document(str(path))
    lines: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            lines.append(t)
    return lines


def pick_side_file(note: str, base_file: str, new_file: str) -> str:
    """
    Decide which firmware file contains the region described by this candidate line.
    """
    m_into = INTO_VER_RE.search(note)
    m_from = FROM_VER_RE.search(note)

    if "insertion" in note.lower():
        # Prefer explicit "into <ver>", otherwise assume it's in the new file
        return new_file
    if "deletion" in note.lower():
        # Prefer explicit "from <ver>", otherwise assume it's in the base file
        return base_file

    # Regions of interest / weaker candidates without clear insertion/deletion:
    # choose new_file as default (you can change to base_file if you prefer)
    return new_file


def parse_candidates(lines: List[str]) -> List[Candidate]:
    current_base = None
    current_new = None
    out: List[Candidate] = []

    for line in lines:
        m_arrow = ARROW_RE.match(line)
        if m_arrow:
            current_base, current_new = m_arrow.group(1), m_arrow.group(2)
            continue

        m_cand = CAND_RE.search(line)
        if m_cand and current_base and current_new:
            start = int(m_cand.group(1))
            end = int(m_cand.group(2))
            note = m_cand.group(3).strip()
            side_file = pick_side_file(note, current_base, current_new)
            out.append(Candidate(start=start, end=end, note=note,
                                 base_file=current_base, new_file=current_new,
                                 side_file=side_file))
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
            c[w[i:i+k]] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to .docx report")
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

    lines = read_docx_lines(report_path)
    candidates = parse_candidates(lines)

    if not candidates:
        print("No candidates found in report. Check formatting.")
        return 2

    # Load needed firmware files
    needed_files = sorted({c.side_file for c in candidates})
    firmwares: Dict[str, bytes] = {}
    missing = []
    for fn in needed_files:
        p = fw_dir / fn
        if not p.exists():
            missing.append(fn)
        else:
            firmwares[fn] = load_firmware_bytes(p)

    if missing:
        print("Missing firmware files in --firmware-dir:")
        for m in missing:
            print(f"  - {m}")
        return 3

    # Collect windows
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

    # Mine patterns
    # Prologue-ish: start_post (bytes immediately after start), and sometimes start_pre
    pro_prefix = mine_prefixes(start_post_windows, args.min_len, args.max_len)
    pro_suffix = mine_suffixes(start_pre_windows, args.min_len, args.max_len)  # "just before entry" markers

    # Epilogue-ish: end_pre (bytes immediately before end), and sometimes end_post
    epi_suffix = mine_suffixes(end_pre_windows, args.min_len, args.max_len)
    epi_prefix = mine_prefixes(end_post_windows, args.min_len, args.max_len)  # "just after exit" markers

    # k-grams in boundary windows
    k_list = [int(x.strip()) for x in args.kgrams.split(",") if x.strip()]
    kgram_reports = {}
    for k in k_list:
        # Focus on the most interesting windows: start_post and end_pre
        c1 = mine_kgrams(start_post_windows, k)
        c2 = mine_kgrams(end_pre_windows, k)
        kgram_reports[k] = {
            "start_post_top": [(bytes_to_hex(b), n) for b, n in c1.most_common(args.top)],
            "end_pre_top": [(bytes_to_hex(b), n) for b, n in c2.most_common(args.top)],
        }

    def top_patterns(counter: Counter) -> List[Tuple[str, int]]:
        # Prefer longer patterns when counts tie: sort by count desc then length desc
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

    # Pretty print headline results
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
