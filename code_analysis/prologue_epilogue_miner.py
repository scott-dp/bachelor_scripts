#!/usr/bin/env python3
"""
Prologue / Epilogue Pattern Miner for Unknown-ISA Firmware
==========================================================
Bachelor thesis tool: mines candidate subroutine boundaries from binary-diffed
Shimano firmware to discover recurring byte patterns that indicate function
prologues, epilogues, and control-flow instructions.

Usage:
    python prologue_epilogue_miner.py --bin-root <path_to_bin_folder> --csv <filtered.csv>

The bin-root folder should contain device subfolders like:
    bin/DUE6001/DUE6001-D.2.8.2.dat
    bin/DUE8000/DUE8000-D.4.0.1.dat
    ...
"""

import argparse
import csv
import os
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION — tune these to experiment
# ──────────────────────────────────────────────────────────────────────
CONTEXT_BYTES = 32        # how many bytes of surrounding context to grab on each side
NGRAM_MIN = 2             # shortest n-gram to consider
NGRAM_MAX = 8             # longest n-gram to consider
BOUNDARY_WINDOW = 16      # bytes at the very start/end of a candidate region to treat as
                          # "prologue zone" or "epilogue zone"
TOP_K = 40                # how many top patterns to show per category
MIN_OCCURRENCES = 3       # minimum times a pattern must appear to be reported
ALIGNMENT_CHECK = [2, 4]  # check if boundary offsets are aligned to these values


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def hex_str(data: bytes) -> str:
    """Pretty hex string."""
    return " ".join(f"{b:02x}" for b in data)


def load_binary(path: str) -> bytes:
    """Load a binary file completely into memory."""
    with open(path, "rb") as f:
        return f.read()


def resolve_file(bin_root: str, device: str, filename: str) -> str | None:
    """Try to find the .dat file under bin_root/device/filename."""
    # Direct path
    p = os.path.join(bin_root, device, filename)
    if os.path.isfile(p):
        return p
    # Some filenames may have spaces (DUE6002 3.1.9.dat); try as-is
    # Also try with dashes replacing spaces
    alt = os.path.join(bin_root, device, filename.replace(" ", "-"))
    if os.path.isfile(alt):
        return alt
    return None


# ──────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────
def extract_boundary_bytes(data: bytes, start: int, end: int, context: int = CONTEXT_BYTES):
    """
    Given a candidate region [start, end) inside `data`, extract:
      - prologue_zone   : bytes at the START of the region  (the first BOUNDARY_WINDOW bytes)
      - epilogue_zone   : bytes at the END of the region    (the last BOUNDARY_WINDOW bytes)
      - pre_context     : bytes just BEFORE the region      (context bytes before start)
      - post_context    : bytes just AFTER the region       (context bytes after end)
      - full_region     : entire candidate region bytes
    All clipped to valid file bounds.
    """
    file_len = len(data)
    # Clip
    s = max(0, start)
    e = min(file_len, end)
    if s >= e:
        return None

    pre_start = max(0, s - context)
    post_end = min(file_len, e + context)

    return {
        "prologue_zone": data[s : min(s + BOUNDARY_WINDOW, e)],
        "epilogue_zone": data[max(s, e - BOUNDARY_WINDOW) : e],
        "pre_context": data[pre_start : s],
        "post_context": data[e : post_end],
        "full_region": data[s : e],
        "start": s,
        "end": e,
    }


def ngrams_from_bytes(data: bytes, n_min: int = NGRAM_MIN, n_max: int = NGRAM_MAX):
    """Yield all (ngram_bytes, offset_within_data) for lengths n_min..n_max."""
    for n in range(n_min, n_max + 1):
        for i in range(len(data) - n + 1):
            yield data[i : i + n], i


# ──────────────────────────────────────────────────────────────────────
# Analysis stages
# ──────────────────────────────────────────────────────────────────────
class PatternMiner:
    def __init__(self):
        # Counters: pattern_bytes -> count
        self.prologue_ngrams = Counter()
        self.epilogue_ngrams = Counter()
        self.pre_context_ngrams = Counter()
        self.post_context_ngrams = Counter()

        # Track which (device, file, offset) contributed each pattern for dedup
        self.prologue_sources = defaultdict(set)
        self.epilogue_sources = defaultdict(set)

        # Raw boundary byte collector (for first/last byte histograms)
        self.first_bytes = Counter()       # first byte of each candidate region
        self.last_bytes = Counter()        # last byte of each candidate region
        self.pre_boundary_bytes = Counter() # byte just before region start
        self.post_boundary_bytes = Counter()# byte just after region end

        # First-2 and Last-2 byte pairs (useful for 16-bit opcode architectures)
        self.first_pairs = Counter()
        self.last_pairs = Counter()

        # Alignment stats
        self.start_offsets = []
        self.end_offsets = []
        self.region_lengths = []

        # Collect full prologue/epilogue zones for cross-correlation
        self.prologue_zones = []
        self.epilogue_zones = []

        self.num_regions = 0
        self.skipped = 0

    def add_region(self, device, filename, bd):
        """Add one extracted boundary dict from extract_boundary_bytes."""
        if bd is None:
            self.skipped += 1
            return
        self.num_regions += 1
        tag = (device, filename, bd["start"])

        # ── Single-byte histograms ──
        region = bd["full_region"]
        if len(region) >= 1:
            self.first_bytes[region[0]] += 1
            self.last_bytes[region[-1]] += 1
        if len(region) >= 2:
            self.first_pairs[region[:2]] += 1
            self.last_pairs[region[-2:]] += 1

        pre = bd["pre_context"]
        post = bd["post_context"]
        if len(pre) >= 1:
            self.pre_boundary_bytes[pre[-1]] += 1
        if len(post) >= 1:
            self.post_boundary_bytes[post[0]] += 1

        # ── N-gram extraction ──
        for ng, _ in ngrams_from_bytes(bd["prologue_zone"]):
            self.prologue_ngrams[ng] += 1
            self.prologue_sources[ng].add(tag)

        for ng, _ in ngrams_from_bytes(bd["epilogue_zone"]):
            self.epilogue_ngrams[ng] += 1
            self.epilogue_sources[ng].add(tag)

        for ng, _ in ngrams_from_bytes(pre):
            self.pre_context_ngrams[ng] += 1

        for ng, _ in ngrams_from_bytes(post):
            self.post_context_ngrams[ng] += 1

        # ── Alignment ──
        self.start_offsets.append(bd["start"])
        self.end_offsets.append(bd["end"])
        self.region_lengths.append(bd["end"] - bd["start"])

        # ── Save zones ──
        self.prologue_zones.append((tag, bd["prologue_zone"]))
        self.epilogue_zones.append((tag, bd["epilogue_zone"]))

    # ── Reporting ──────────────────────────────────────────────────

    def _top_patterns(self, counter, top_k=TOP_K, min_occ=MIN_OCCURRENCES, source_map=None):
        """Return list of (hex_string, count, length, num_unique_sources) tuples."""
        results = []
        for pat, cnt in counter.most_common(top_k * 5):  # oversample then filter
            if cnt < min_occ:
                break
            n_sources = len(source_map[pat]) if source_map and pat in source_map else None
            results.append((hex_str(pat), cnt, len(pat), n_sources))
            if len(results) >= top_k:
                break
        return results

    def _byte_histogram(self, counter, label, top_n=20):
        lines = [f"\n{'='*60}", f"  {label}  (top {top_n})", f"{'='*60}"]
        for val, cnt in counter.most_common(top_n):
            if isinstance(val, int):
                lines.append(f"  0x{val:02x}  ({val:3d})  :  {cnt:4d} times  "
                             f"({'#' * min(cnt, 60)}")
            else:
                lines.append(f"  {hex_str(val):14s}  :  {cnt:4d} times  "
                             f"({'#' * min(cnt, 60)}")
        return "\n".join(lines)

    def _pattern_table(self, patterns, label):
        lines = [f"\n{'='*70}", f"  {label}", f"{'='*70}"]
        lines.append(f"  {'Pattern':<30s}  {'Cnt':>5s}  {'Len':>3s}  {'Srcs':>5s}")
        lines.append(f"  {'-'*30}  {'-'*5}  {'-'*3}  {'-'*5}")
        for hexs, cnt, length, srcs in patterns:
            src_str = str(srcs) if srcs is not None else "—"
            lines.append(f"  {hexs:<30s}  {cnt:5d}  {length:3d}  {src_str:>5s}")
        return "\n".join(lines)

    def alignment_report(self):
        lines = [f"\n{'='*60}", "  ALIGNMENT & LENGTH ANALYSIS", f"{'='*60}"]
        for align in ALIGNMENT_CHECK:
            starts_aligned = sum(1 for o in self.start_offsets if o % align == 0)
            ends_aligned = sum(1 for o in self.end_offsets if o % align == 0)
            lens_aligned = sum(1 for l in self.region_lengths if l % align == 0)
            n = len(self.start_offsets)
            lines.append(f"  Alignment to {align}: starts={starts_aligned}/{n} "
                         f"({100*starts_aligned/n:.0f}%)  "
                         f"ends={ends_aligned}/{n} ({100*ends_aligned/n:.0f}%)  "
                         f"lengths={lens_aligned}/{n} ({100*lens_aligned/n:.0f}%)")

        if self.region_lengths:
            import statistics
            lines.append(f"\n  Region length stats:")
            lines.append(f"    min={min(self.region_lengths)}  max={max(self.region_lengths)}  "
                         f"mean={statistics.mean(self.region_lengths):.1f}  "
                         f"median={statistics.median(self.region_lengths):.1f}")

            # Length modulo distribution (hints at instruction width)
            lines.append(f"\n  Region length mod distribution (hints at instruction width):")
            for mod in [2, 3, 4]:
                dist = Counter(l % mod for l in self.region_lengths)
                lines.append(f"    mod {mod}: {dict(sorted(dist.items()))}")

        return "\n".join(lines)

    def cross_correlation_report(self):
        """Find prologue zones that share long common substrings across different devices."""
        lines = [f"\n{'='*60}", "  CROSS-DEVICE PATTERN MATCHES", f"{'='*60}"]
        lines.append("  (Prologue zones from different devices sharing ≥4 byte subsequences)\n")

        # Group by device
        by_device = defaultdict(list)
        for (tag, zone) in self.prologue_zones:
            by_device[tag[0]].append((tag, zone))

        devices = sorted(by_device.keys())
        cross_matches = Counter()

        for i, d1 in enumerate(devices):
            for d2 in devices[i + 1:]:
                for (tag1, z1) in by_device[d1]:
                    for (tag2, z2) in by_device[d2]:
                        # Find common 4-byte subsequences
                        set1 = set()
                        for n in range(4, min(len(z1), len(z2), 9)):
                            for k in range(len(z1) - n + 1):
                                set1.add(z1[k:k+n])
                        for n in range(4, min(len(z1), len(z2), 9)):
                            for k in range(len(z2) - n + 1):
                                sub = z2[k:k+n]
                                if sub in set1:
                                    cross_matches[sub] += 1

        if cross_matches:
            # Filter to the longest non-overlapping patterns
            top = cross_matches.most_common(30)
            for pat, cnt in top:
                lines.append(f"  {hex_str(pat):<30s}  matched {cnt} times across devices  (len={len(pat)})")
        else:
            lines.append("  No cross-device matches found (4+ bytes).")

        return "\n".join(lines)

    def repeated_fixed_bytes_report(self):
        """Look for runs of the same byte (e.g., 0x00 or 0xFF padding) at boundaries."""
        lines = [f"\n{'='*60}", "  PADDING / NOP SLED DETECTION AT BOUNDARIES", f"{'='*60}"]

        padding_before = Counter()  # what repeated-byte padding appears before regions
        padding_after = Counter()

        for (tag, zone) in self.prologue_zones:
            # Check first few bytes for repeats
            if len(zone) >= 4:
                if zone[0] == zone[1] == zone[2] == zone[3]:
                    padding_before[zone[0]] += 1

        for (tag, zone) in self.epilogue_zones:
            if len(zone) >= 4:
                if zone[-1] == zone[-2] == zone[-3] == zone[-4]:
                    padding_after[zone[-1]] += 1

        lines.append("  Repeated-byte runs at START of regions:")
        for b, cnt in padding_before.most_common(10):
            lines.append(f"    0x{b:02x} repeated: {cnt} regions")

        lines.append("  Repeated-byte runs at END of regions:")
        for b, cnt in padding_after.most_common(10):
            lines.append(f"    0x{b:02x} repeated: {cnt} regions")

        return "\n".join(lines)

    def generate_full_report(self):
        sep = "\n" + "=" * 70

        report = []
        report.append(sep)
        report.append("  FIRMWARE PROLOGUE / EPILOGUE PATTERN MINING REPORT")
        report.append(f"  Regions analyzed: {self.num_regions}  |  Skipped: {self.skipped}")
        report.append(sep)

        # 1. Single-byte histograms
        report.append(self._byte_histogram(self.first_bytes, "FIRST BYTE of each candidate region"))
        report.append(self._byte_histogram(self.last_bytes, "LAST BYTE of each candidate region"))
        report.append(self._byte_histogram(self.pre_boundary_bytes, "BYTE just BEFORE region start (potential end-of-previous-func)"))
        report.append(self._byte_histogram(self.post_boundary_bytes, "BYTE just AFTER region end"))

        # 2. Byte-pair histograms (16-bit)
        report.append(self._byte_histogram(self.first_pairs, "FIRST 2 BYTES (potential opcode) of each region"))
        report.append(self._byte_histogram(self.last_pairs, "LAST 2 BYTES (potential return opcode) of each region"))

        # 3. N-gram patterns
        report.append(self._pattern_table(
            self._top_patterns(self.prologue_ngrams, source_map=self.prologue_sources),
            "TOP PROLOGUE N-GRAMS (first 16 bytes of each region)"))

        report.append(self._pattern_table(
            self._top_patterns(self.epilogue_ngrams, source_map=self.epilogue_sources),
            "TOP EPILOGUE N-GRAMS (last 16 bytes of each region)"))

        report.append(self._pattern_table(
            self._top_patterns(self.pre_context_ngrams),
            "TOP PRE-CONTEXT N-GRAMS (32 bytes before region — may contain previous func epilogue)"))

        report.append(self._pattern_table(
            self._top_patterns(self.post_context_ngrams),
            "TOP POST-CONTEXT N-GRAMS (32 bytes after region — may contain next func prologue)"))

        # 4. Alignment
        report.append(self.alignment_report())

        # 5. Cross-device
        report.append(self.cross_correlation_report())

        # 6. Padding
        report.append(self.repeated_fixed_bytes_report())

        return "\n".join(report)


# ──────────────────────────────────────────────────────────────────────
# Detailed per-region dump (for manual inspection)
# ──────────────────────────────────────────────────────────────────────
def dump_region_detail(bd, device, filename, region_idx, f_out):
    """Write a detailed hex dump of one region and its context to a file."""
    f_out.write(f"\n{'─'*70}\n")
    f_out.write(f"Region #{region_idx}  |  {device} / {filename}  |  "
                f"offset 0x{bd['start']:06x}–0x{bd['end']:06x}  "
                f"({bd['end'] - bd['start']} bytes)\n")
    f_out.write(f"{'─'*70}\n")

    f_out.write(f"  PRE-CONTEXT  ({len(bd['pre_context'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['pre_context'])}\n")
    f_out.write(f"  >>>  PROLOGUE ZONE  ({len(bd['prologue_zone'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['prologue_zone'])}\n")

    full = bd['full_region']
    if len(full) > 2 * BOUNDARY_WINDOW:
        mid_start = BOUNDARY_WINDOW
        mid_end = len(full) - BOUNDARY_WINDOW
        f_out.write(f"  ... middle {mid_end - mid_start} bytes omitted ...\n")

    f_out.write(f"  <<<  EPILOGUE ZONE  ({len(bd['epilogue_zone'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['epilogue_zone'])}\n")
    f_out.write(f"  POST-CONTEXT ({len(bd['post_context'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['post_context'])}\n")


# ──────────────────────────────────────────────────────────────────────
# Byte-pair entropy (to gauge whether regions look like code vs data)
# ──────────────────────────────────────────────────────────────────────
def byte_entropy_report(miner):
    """Compute byte-level entropy of collected prologue vs epilogue zones."""
    import math
    lines = [f"\n{'='*60}", "  BYTE ENTROPY ANALYSIS", f"{'='*60}"]

    for label, zones in [("Prologue zones", miner.prologue_zones),
                         ("Epilogue zones", miner.epilogue_zones)]:
        all_bytes = b"".join(z for _, z in zones)
        if not all_bytes:
            continue
        freq = Counter(all_bytes)
        total = len(all_bytes)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        lines.append(f"  {label}: {total} bytes, Shannon entropy = {entropy:.3f} bits/byte")
        lines.append(f"    (random=8.0, English text≈4.5, typical code≈5-7, compressed≈7.5+)")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    global CONTEXT_BYTES, BOUNDARY_WINDOW, NGRAM_MAX, MIN_OCCURRENCES
    parser = argparse.ArgumentParser(
        description="Mine prologue/epilogue patterns from firmware candidate regions")
    parser.add_argument("--bin-root", required=True,
                        help="Root directory containing device subfolders with .dat files")
    parser.add_argument("--csv", required=True,
                        help="Path to filtered.csv with candidate regions")
    parser.add_argument("--output", default="pattern_report.txt",
                        help="Output report file (default: pattern_report.txt)")
    parser.add_argument("--dump", default="region_details.txt",
                        help="Detailed per-region hex dump file (default: region_details.txt)")
    parser.add_argument("--json", default="patterns.json",
                        help="Machine-readable JSON output (default: patterns.json)")
    parser.add_argument("--context", type=int, default=CONTEXT_BYTES,
                        help=f"Bytes of context around each region (default: {CONTEXT_BYTES})")
    parser.add_argument("--window", type=int, default=BOUNDARY_WINDOW,
                        help=f"Prologue/epilogue zone size in bytes (default: {BOUNDARY_WINDOW})")
    parser.add_argument("--ngram-max", type=int, default=NGRAM_MAX,
                        help=f"Max n-gram length (default: {NGRAM_MAX})")
    parser.add_argument("--min-occ", type=int, default=MIN_OCCURRENCES,
                        help=f"Minimum occurrences to report (default: {MIN_OCCURRENCES})")
    args = parser.parse_args()


    CONTEXT_BYTES = args.context
    BOUNDARY_WINDOW = args.window
    NGRAM_MAX = args.ngram_max
    MIN_OCCURRENCES = args.min_occ

    bin_root = args.bin_root
    if not os.path.isdir(bin_root):
        print(f"ERROR: bin-root '{bin_root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # ── Load CSV ──
    rows = []
    with open(args.csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} candidate regions from {args.csv}")

    # ── Cache loaded binaries ──
    file_cache: dict[str, bytes] = {}

    def get_binary(device, filename):
        key = f"{device}/{filename}"
        if key not in file_cache:
            path = resolve_file(bin_root, device, filename)
            if path is None:
                print(f"  WARNING: could not find {key}", file=sys.stderr)
                file_cache[key] = None
            else:
                file_cache[key] = load_binary(path)
                print(f"  Loaded {key} ({len(file_cache[key])} bytes)")
        return file_cache[key]

    # ── Process each row ──
    miner = PatternMiner()

    detail_file = open(args.dump, "w", encoding="utf-8")
    detail_file.write("DETAILED REGION HEX DUMPS\n")
    detail_file.write(f"Generated by prologue_epilogue_miner.py\n\n")

    for i, row in enumerate(rows):
        device = row["device"].strip()
        side = row.get("side", "").strip()
        start = int(row["start"])
        end = int(row["end"])

        # Determine which file to read from
        if side == "new":
            filename = row["new_file"].strip()
        elif side == "base":
            filename = row["base_file"].strip()
        else:
            # If side is empty, default to new_file (the "insertion" heuristic)
            filename = row["new_file"].strip()

        data = get_binary(device, filename)
        if data is None:
            miner.skipped += 1
            continue

        bd = extract_boundary_bytes(data, start, end, CONTEXT_BYTES)
        miner.add_region(device, filename, bd)

        if bd is not None:
            dump_region_detail(bd, device, filename, i, detail_file)

    detail_file.close()
    print(f"\nProcessed {miner.num_regions} regions, skipped {miner.skipped}")

    # ── Generate report ──
    report = miner.generate_full_report()
    report += byte_entropy_report(miner)

    # Write text report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")
    print(f"Detail dump written to {args.dump}")

    # ── JSON export ──
    json_data = {
        "summary": {
            "regions_analyzed": miner.num_regions,
            "regions_skipped": miner.skipped,
        },
        "top_prologue_ngrams": [
            {"pattern": h, "count": c, "length": l, "unique_sources": s}
            for h, c, l, s in miner._top_patterns(
                miner.prologue_ngrams, source_map=miner.prologue_sources)
        ],
        "top_epilogue_ngrams": [
            {"pattern": h, "count": c, "length": l, "unique_sources": s}
            for h, c, l, s in miner._top_patterns(
                miner.epilogue_ngrams, source_map=miner.epilogue_sources)
        ],
        "first_byte_histogram": {
            f"0x{b:02x}": cnt for b, cnt in miner.first_bytes.most_common(20)
        },
        "last_byte_histogram": {
            f"0x{b:02x}": cnt for b, cnt in miner.last_bytes.most_common(20)
        },
        "first_pair_histogram": {
            hex_str(p): cnt for p, cnt in miner.first_pairs.most_common(20)
        },
        "last_pair_histogram": {
            hex_str(p): cnt for p, cnt in miner.last_pairs.most_common(20)
        },
        "alignment": {},
    }
    for align in ALIGNMENT_CHECK:
        n = len(miner.start_offsets)
        if n > 0:
            json_data["alignment"][f"mod_{align}"] = {
                "starts_aligned": sum(1 for o in miner.start_offsets if o % align == 0),
                "ends_aligned": sum(1 for o in miner.end_offsets if o % align == 0),
                "lengths_aligned": sum(1 for l in miner.region_lengths if l % align == 0),
                "total": n,
            }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data written to {args.json}")

    # ── Print summary to console ──
    print("\n" + "=" * 70)
    print("  QUICK SUMMARY")
    print("=" * 70)

    print("\n  Top 10 PROLOGUE patterns (first bytes of candidate regions):")
    for h, c, l, s in miner._top_patterns(miner.prologue_ngrams,
                                            source_map=miner.prologue_sources)[:10]:
        print(f"    {h:<24s}  count={c:3d}  len={l}  sources={s}")

    print("\n  Top 10 EPILOGUE patterns (last bytes of candidate regions):")
    for h, c, l, s in miner._top_patterns(miner.epilogue_ngrams,
                                            source_map=miner.epilogue_sources)[:10]:
        print(f"    {h:<24s}  count={c:3d}  len={l}  sources={s}")

    print("\n  Top 5 FIRST byte-pairs (potential call/prologue opcodes):")
    for p, cnt in miner.first_pairs.most_common(5):
        print(f"    {hex_str(p)}  :  {cnt} times")

    print("\n  Top 5 LAST byte-pairs (potential return opcodes):")
    for p, cnt in miner.last_pairs.most_common(5):
        print(f"    {hex_str(p)}  :  {cnt} times")

    print(f"\n  Full report: {args.output}")
    print(f"  Hex dumps:   {args.dump}")
    print(f"  JSON data:   {args.json}")


if __name__ == "__main__":
    main()
