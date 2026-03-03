import argparse
import csv
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set


# A pattern is a tuple of length k: each entry is an int 0..255, or None (wildcard)
Pattern = Tuple[Optional[int], ...]


def pattern_to_hex(pat: Pattern) -> str:
    return " ".join("??" if b is None else f"{b:02x}" for b in pat)


def slice_window(data: bytes, center: int, window: int) -> Tuple[int, bytes]:
    """Returns (window_start_offset_in_file, window_bytes). Window is [center-window, center+window)."""
    start = max(0, center - window)
    end = min(len(data), center + window)
    return start, data[start:end]


def generate_masked_patterns(kgram: bytes, wildcard_max: int) -> List[Pattern]:
    """Generate patterns by replacing up to wildcard_max positions with None (wildcard)."""
    k = len(kgram)
    base: Pattern = tuple(kgram[i] for i in range(k))

    patterns: List[Pattern] = [base]  # 0 wildcards
    if wildcard_max <= 0:
        return patterns

    positions = range(k)
    for w in range(1, min(wildcard_max, k) + 1):
        for combo in itertools.combinations(positions, w):
            pat = list(base)
            for idx in combo:
                pat[idx] = None
            patterns.append(tuple(pat))

    return patterns


@dataclass
class Occurrence:
    device: str
    file_path: str
    file_role: str  # "base" or "new"
    boundary_kind: str  # "start" or "end"
    boundary_offset: int
    window_start: int
    kgram_offset: int  # absolute offset in file where the k-gram starts


def resolve_file(bin_root: str, device: str, filename: str) -> Optional[str]:
    # Typical layout: ..\bin\<device>\<filename>
    candidate = os.path.join(bin_root, device, filename)
    if os.path.isfile(candidate):
        return candidate

    # Fallback: brute-search within device folder (handles odd subfolders / names)
    dev_dir = os.path.join(bin_root, device)
    if os.path.isdir(dev_dir):
        for root, _, files in os.walk(dev_dir):
            for f in files:
                if f == filename:
                    p = os.path.join(root, f)
                    if os.path.isfile(p):
                        return p
    return None


def load_bytes_cached(path: str, cache: Dict[str, bytes]) -> bytes:
    if path not in cache:
        with open(path, "rb") as f:
            cache[path] = f.read()
    return cache[path]


def mine_patterns(
    csv_path: str,
    bin_root: str,
    window: int,
    k: int,
    wildcard_max: int,
    min_total_hits: int,
    min_files: int,
    top_n: int,
):
    total_hits: Dict[Pattern, int] = defaultdict(int)
    files_seen: Dict[Pattern, Set[str]] = defaultdict(set)
    examples: Dict[Pattern, List[Occurrence]] = defaultdict(list)

    file_cache: Dict[str, bytes] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"device", "base_file", "new_file", "base_version", "new_version", "start", "end", "len", "side"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

        for row_i, row in enumerate(reader, start=1):
            device = row["device"].strip()
            base_file = row["base_file"].strip()
            new_file = row["new_file"].strip()

            side = (row.get("side") or "").strip().lower()
            if side not in {"base", "new"}:
                # skip rows without a usable side
                continue

            try:
                start = int(row["start"])
                end = int(row["end"])
            except ValueError:
                continue

            # Pick the correct file based on side
            file_role = side
            fname = base_file if side == "base" else new_file

            path = resolve_file(bin_root, device, fname)
            if not path:
                continue

            data = load_bytes_cached(path, file_cache)

            # Examine both boundaries (start and end) in that file
            for boundary_kind, boundary_offset in (("start", start), ("end", end)):
                if boundary_offset < 0 or boundary_offset >= len(data):
                    continue

                win_start, win_bytes = slice_window(data, boundary_offset, window)
                if len(win_bytes) < k:
                    continue

                for local_i in range(0, len(win_bytes) - k + 1):
                    kgram = win_bytes[local_i : local_i + k]
                    abs_kgram_offset = win_start + local_i

                    for pat in generate_masked_patterns(kgram, wildcard_max):
                        total_hits[pat] += 1
                        files_seen[pat].add(path)

                        if len(examples[pat]) < 8:
                            examples[pat].append(
                                Occurrence(
                                    device=device,
                                    file_path=path,
                                    file_role=file_role,
                                    boundary_kind=boundary_kind,
                                    boundary_offset=boundary_offset,
                                    window_start=win_start,
                                    kgram_offset=abs_kgram_offset,
                                )
                            )

    rows = []
    for pat, hits in total_hits.items():
        fcount = len(files_seen[pat])
        if hits >= min_total_hits and fcount >= min_files:
            rows.append((hits, fcount, pat))

    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    rows = rows[:top_n]

    return rows, files_seen, examples


def write_reports(rows, files_seen, examples, out_report, out_occ):
    with open(out_report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "total_hits_in_windows", "file_count", "files", "pattern_hex"])
        for rank, (hits, fcount, pat) in enumerate(rows, start=1):
            files_list = sorted(files_seen[pat])
            w.writerow([rank, hits, fcount, ";".join(files_list), pattern_to_hex(pat)])

    with open(out_occ, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "pattern_hex",
                "device",
                "file_path",
                "file_role",
                "boundary_kind",
                "boundary_offset",
                "window_start",
                "kgram_offset",
            ]
        )
        for rank, (_, _, pat) in enumerate(rows, start=1):
            pat_hex = pattern_to_hex(pat)
            for occ in examples.get(pat, []):
                w.writerow(
                    [
                        rank,
                        pat_hex,
                        occ.device,
                        occ.file_path,
                        occ.file_role,
                        occ.boundary_kind,
                        occ.boundary_offset,
                        occ.window_start,
                        occ.kgram_offset,
                    ]
                )


def main():
    ap = argparse.ArgumentParser(
        description="Mine common k-gram (with wildcards) patterns around boundary windows from CSV (uses 'side' to pick file)"
    )
    ap.add_argument("--csv", default="filtered.csv", help="Path to CSV (default: filtered.csv)")
    ap.add_argument("--bin-root", default=os.path.join("..", "bin"), help="Path to bin folder (default: ..\\bin)")
    ap.add_argument("--window", type=int, default=96, help="Half-window size around boundary (bytes)")
    ap.add_argument("--k", type=int, default=12, help="k-gram length (bytes)")
    ap.add_argument("--wildcards", type=int, default=2, help="Max wildcard bytes per pattern (??)")
    ap.add_argument("--min-hits", type=int, default=30, help="Minimum total hits across all windows")
    ap.add_argument("--min-files", type=int, default=5, help="Minimum distinct files a pattern must appear in")
    ap.add_argument("--top", type=int, default=200, help="How many top patterns to output")
    ap.add_argument("--out-report", default="patterns_report.csv", help="Output summary report CSV")
    ap.add_argument("--out-occ", default="pattern_occurrences.csv", help="Output occurrences CSV (examples for top patterns)")

    args = ap.parse_args()

    rows, files_seen, examples = mine_patterns(
        csv_path=args.csv,
        bin_root=args.bin_root,
        window=args.window,
        k=args.k,
        wildcard_max=args.wildcards,
        min_total_hits=args.min_hits,
        min_files=args.min_files,
        top_n=args.top,
    )

    write_reports(rows, files_seen, examples, args.out_report, args.out_occ)

    print("Done.")
    print(f"  Wrote: {args.out_report}")
    print(f"  Wrote: {args.out_occ}")
    if rows:
        hits, fcount, pat = rows[0]
        print(f"Top pattern: hits={hits}, files={fcount}")
        print(f"  {pattern_to_hex(pat)}")
    else:
        print("No patterns met thresholds. Try lowering --min-hits/--min-files, or using smaller k / fewer wildcards.")


if __name__ == "__main__":
    main()
