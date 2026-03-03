#!/usr/bin/env python3
"""
Prologue / Epilogue Pattern Miner v2 — Wildcard-Aware Edition
==============================================================
Bachelor thesis tool for mining subroutine boundaries from binary-diffed
Shimano firmware.  Discovers recurring byte patterns (with wildcard/variable
byte support) that indicate function prologues, epilogues, and control-flow
instructions.

Key improvements over v1:
  • Wildcard-aware pattern mining — detects opcode+immediate patterns where
    only the opcode byte(s) are fixed and operands vary.
  • Positional byte entropy — shows which byte positions in the prologue /
    epilogue zones are "fixed" (low entropy = likely opcode) vs "variable"
    (high entropy = likely operand / immediate).
  • Template extraction — clusters similar byte sequences and extracts the
    fixed skeleton with ?? wildcards for variable positions.
  • ISA-informed heuristic scanning — scans for byte patterns known to be
    prologues/epilogues on common embedded MCU architectures (Renesas RL78,
    RX, SH-2, ARM Thumb, 8051, AVR, PIC).
  • Full source tracking — every reported pattern shows exactly which files
    and byte offsets it was found in.

Usage:
    python prologue_epilogue_miner_v2.py \\
        --bin-root <path_to_bin_folder> \\
        --csv filtered.csv

Output files:
    pattern_report_v2.txt   — full human-readable report
    region_details_v2.txt   — per-region hex dumps with context
    patterns_v2.json        — machine-readable JSON for further scripting
"""

import argparse
import csv
import math
import os
import sys
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CONTEXT_BYTES       = 32    # surrounding context on each side of a region
BOUNDARY_WINDOW     = 16    # bytes at start/end treated as prologue/epilogue zone
NGRAM_MIN           = 2     # shortest exact n-gram
NGRAM_MAX           = 8     # longest exact n-gram
TOP_K               = 40    # top patterns per category
MIN_OCCURRENCES     = 3     # minimum hits to report
ALIGNMENT_CHECK     = [2, 4]
WILDCARD_LENGTHS    = [2, 3, 4, 5, 6]  # template lengths to try
ENTROPY_THRESHOLD   = 3.0   # bits — above this a byte position is "variable"
TEMPLATE_MIN_FIXED  = 1     # a template must have at least this many fixed bytes
TEMPLATE_MIN_OCC    = 3     # minimum occurrences to report a wildcard template


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def hex_str(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)

def template_str(template: list) -> str:
    """Format a template where None = wildcard."""
    return " ".join(f"{b:02x}" if b is not None else "??" for b in template)

def load_binary(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def resolve_file(bin_root: str, device: str, filename: str) -> str | None:
    p = os.path.join(bin_root, device, filename)
    if os.path.isfile(p):
        return p
    alt = os.path.join(bin_root, device, filename.replace(" ", "-"))
    if os.path.isfile(alt):
        return alt
    return None

def byte_entropy(counts: Counter, total: int) -> float:
    """Shannon entropy in bits from a Counter of byte values."""
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

def source_tag(device, filename, offset):
    """Short human-readable source identifier."""
    # Strip common prefix/suffix for readability
    name = filename.replace(".dat", "")
    return f"{device}/{name}@0x{offset:05x}"


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_boundary_bytes(data: bytes, start: int, end: int, context: int = CONTEXT_BYTES):
    file_len = len(data)
    s = max(0, start)
    e = min(file_len, end)
    if s >= e:
        return None

    pre_start = max(0, s - context)
    post_end = min(file_len, e + context)

    return {
        "prologue_zone": data[s : min(s + BOUNDARY_WINDOW, e)],
        "epilogue_zone": data[max(s, e - BOUNDARY_WINDOW) : e],
        "pre_context":   data[pre_start : s],
        "post_context":  data[e : post_end],
        "full_region":   data[s : e],
        "start": s,
        "end": e,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ISA heuristic signatures
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: (name, description, byte_pattern, mask)
# mask: list of bools, True = must match exactly, False = wildcard
# We scan the prologue/epilogue zones for these.
#
# These cover the most common embedded MCU families found in bike electronics.
# The key idea: we check ALL of them and see which ones actually hit.  Whatever
# matches consistently is a strong clue about the ISA.

ISA_SIGNATURES = []

def sig(name, desc, pattern_hex, mask_hex=None):
    """Register an ISA heuristic signature."""
    pat = bytes.fromhex(pattern_hex.replace(" ", ""))
    if mask_hex:
        mask = bytes.fromhex(mask_hex.replace(" ", ""))
        m = [b == 0xFF for b in mask]
    else:
        m = [True] * len(pat)
    ISA_SIGNATURES.append((name, desc, pat, m))

# ── ARM Thumb (16-bit, little-endian, common in Cortex-M) ──────────────
sig("ARM-Thumb PUSH {LR}",
    "push {lr} — standard function prologue",
    "00 B5",                  # PUSH {LR} = 0xB500 in LE → bytes 00 B5
    "00 FF")                  # low byte varies with register list
sig("ARM-Thumb PUSH {r4-r7,LR}",
    "push {r4-r7,lr} — callee-saved regs + LR",
    "F0 B5")                  # exact
sig("ARM-Thumb PUSH {r4,LR}",
    "push {r4,lr}",
    "10 B5")
sig("ARM-Thumb PUSH generic",
    "push {reglist, lr} — any PUSH with LR set",
    "00 B5",
    "00 FF")                  # mask: high byte must be B5, low byte varies
sig("ARM-Thumb POP {PC}",
    "pop {pc} — standard return (epilogue)",
    "00 BD",
    "00 FF")
sig("ARM-Thumb BX LR",
    "bx lr — return via link register",
    "70 47")
sig("ARM-Thumb SUB SP",
    "sub sp, #imm — allocate stack frame",
    "00 B0",
    "80 FF")                  # bit 7 of low byte = 1 for SUB, 0 for ADD
sig("ARM-Thumb BL (high)",
    "bl <target> — branch with link (call), high halfword",
    "00 F0",
    "00 F8")                  # F0-F7 in high byte
sig("ARM-Thumb BL (low)",
    "bl <target> — branch with link (call), low halfword",
    "00 F8",
    "00 F8")                  # F800-FFFF range

# ── Renesas RL78 (8/16-bit, very common in Shimano hardware) ───────────
sig("RL78 PUSH AX",       "push ax", "C1")
sig("RL78 PUSH BC",       "push bc", "C5")
sig("RL78 PUSH DE",       "push de", "C3")
sig("RL78 PUSH HL",       "push hl", "C7")
sig("RL78 POP AX",        "pop ax",  "C0")
sig("RL78 POP BC",        "pop bc",  "C4")
sig("RL78 POP DE",        "pop de",  "C2")
sig("RL78 POP HL",        "pop hl",  "C6")
sig("RL78 RET",           "ret — return from subroutine", "D7")
sig("RL78 RETI",          "reti — return from interrupt", "61 FC")
sig("RL78 CALL !addr16",  "call !addr16 — 16-bit direct call", "9A 00 00", "FF 00 00")
sig("RL78 CALL $addr20",  "call $addr20 — 20-bit call",       "FC 00 00", "FF 00 00")
sig("RL78 CALLT",         "callt — table call", "C1",  "C1")  # C1 overlap w/ PUSH AX
sig("RL78 BR !addr16",    "br !addr16 — unconditional branch", "9B 00 00", "FF 00 00")
sig("RL78 BR $addr20",    "br $addr20 — 20-bit branch",       "FE 00 00 00", "FF 00 00 00")
sig("RL78 MOVW SP,#imm",  "movw sp, #imm16 — set stack pointer", "CB F8 00 00", "FF FF 00 00")
sig("RL78 SUB SP,#imm",   "subw sp, #imm8 — allocate stack frame (4th map)", "DA 00", "FF 00")
sig("RL78 NOP",           "nop", "00")

# ── Renesas RX (32-bit, little-endian) ─────────────────────────────────
sig("RX PUSHM",  "pushm — push multiple registers", "6E 00", "FF 00")
sig("RX POPM",   "popm — pop multiple registers",   "6F 00", "FF 00")
sig("RX RTS",    "rts — return from subroutine",    "02")
sig("RX BSR.W",  "bsr.w — branch to subroutine",    "39 00 00", "FF 00 00")
sig("RX SUB SP", "sub #imm, sp",                    "71 00 00", "FF 00 00")

# ── Renesas SH-2 / SH-2A (16-bit insns, LE) ──────────────────────────
sig("SH2 MOV.L Rm,@-R15", "push register to stack",  "2F 00", "FF 0F")  # 2Fn6
sig("SH2 MOV.L @R15+,Rm", "pop register from stack",  "6F 00", "FF 0F")  # 6Fn6
sig("SH2 RTS",            "rts — return",              "0B 00")
sig("SH2 JSR @Rm",        "jsr — jump to subroutine",  "0B 40", "0F F0")
sig("SH2 BSR disp",       "bsr — branch to subroutine","00 B0", "00 FF")

# ── Intel 8051 ─────────────────────────────────────────────────────────
sig("8051 PUSH direct", "push direct byte", "C0 00", "FF 00")
sig("8051 POP direct",  "pop direct byte",  "D0 00", "FF 00")
sig("8051 RET",         "ret",              "22")
sig("8051 RETI",        "reti",             "32")
sig("8051 LCALL",       "lcall addr16",     "12 00 00", "FF 00 00")
sig("8051 ACALL",       "acall addr11",     "11", "1F")   # xxx10001
sig("8051 LJMP",        "ljmp addr16",      "02 00 00", "FF 00 00")
sig("8051 NOP",         "nop",              "00")

# ── AVR (16-bit insns, LE) ────────────────────────────────────────────
sig("AVR PUSH Rr",  "push register",       "0F 92", "0F FE")
sig("AVR POP Rd",   "pop register",        "0F 90", "0F FE")
sig("AVR RET",      "ret",                 "08 95")
sig("AVR RETI",     "reti",                "18 95")
sig("AVR RCALL",    "rcall — relative call","00 D0", "00 F0")
sig("AVR CALL",     "call — long call",     "0E 94", "0F FE")

# ── PIC18 (common in motor controllers) ───────────────────────────────
sig("PIC18 CALL",   "call addr",  "00 EC", "00 FF")
sig("PIC18 RETURN", "return",     "12 00")
sig("PIC18 RETFIE", "retfie",     "10 00")

# ── Generic patterns (architecture-agnostic) ──────────────────────────
sig("Generic 0xFF padding", "FF padding between functions", "FF FF FF FF")
sig("Generic 0x00 padding", "00 padding / NOP sled",        "00 00 00 00")
sig("Generic 0xCC padding", "CC padding (INT3 / fill byte)","CC CC CC CC")


def match_signature(data: bytes, offset: int, pattern: bytes, mask: list[bool]) -> bool:
    """Check if `pattern` matches `data` at `offset`, respecting wildcard mask."""
    if offset + len(pattern) > len(data):
        return False
    for i, (p, m) in enumerate(zip(pattern, mask)):
        if m and data[offset + i] != p:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Wildcard Template Miner
# ──────────────────────────────────────────────────────────────────────────────
class WildcardTemplateMiner:
    """
    Given a collection of byte sequences (e.g. all prologue zones), finds
    recurring patterns where some byte positions are fixed (opcode) and
    others are variable (operand/immediate).

    Method:
    1. For each template length L, slide a window of size L across all zones.
    2. Group windows by their "signature" — the values at positions that have
       low entropy (< threshold) across many windows.
    3. Positions with high entropy become wildcards (??).
    4. Report templates that occur above min_occ threshold.
    """

    def __init__(self, zones, labels, min_occ=TEMPLATE_MIN_OCC,
                 entropy_threshold=ENTROPY_THRESHOLD):
        self.zones = zones        # list of bytes
        self.labels = labels      # parallel list of source tags
        self.min_occ = min_occ
        self.entropy_thresh = entropy_threshold

    def mine_templates(self, lengths=WILDCARD_LENGTHS):
        """
        Returns list of dicts:
          { "template": [0x3F, None, None, 0x02], "count": N,
            "sources": [...], "fixed_ratio": 0.5 }
        None = wildcard byte.
        """
        all_templates = []

        for L in lengths:
            # Collect all L-byte windows with their source tags
            windows = []  # (bytes, source_tag)
            for zone, label in zip(self.zones, self.labels):
                for i in range(len(zone) - L + 1):
                    windows.append((zone[i:i+L], label, i))

            if len(windows) < self.min_occ:
                continue

            # ── Step 1: Per-position entropy across ALL windows ──
            # This is a rough guide; we refine per-cluster below
            pos_counters = [Counter() for _ in range(L)]
            for (w, _, _) in windows:
                for j in range(L):
                    pos_counters[j][w[j]] += 1

            total_w = len(windows)
            pos_entropy = [byte_entropy(c, total_w) for c in pos_counters]

            # Identify "anchor" positions = low entropy
            anchor_positions = [j for j in range(L) if pos_entropy[j] < self.entropy_thresh]

            if len(anchor_positions) < TEMPLATE_MIN_FIXED:
                continue  # no stable bytes at this length, skip

            # ── Step 2: Group by anchor byte values ──
            groups = defaultdict(list)  # anchor_key -> list of (window, source, local_off)
            for (w, src, off) in windows:
                key = tuple(w[j] for j in anchor_positions)
                groups[key].append((w, src, off))

            # ── Step 3: For each group, refine which positions are truly fixed ──
            for key, members in groups.items():
                if len(members) < self.min_occ:
                    continue

                # Recompute per-position entropy within this cluster
                cluster_counters = [Counter() for _ in range(L)]
                for (w, _, _) in members:
                    for j in range(L):
                        cluster_counters[j][w[j]] += 1

                cluster_size = len(members)
                template = []
                n_fixed = 0
                for j in range(L):
                    ent = byte_entropy(cluster_counters[j], cluster_size)
                    if ent < 1.0:  # very low entropy within cluster = fixed
                        # Use the most common value
                        most_common_val = cluster_counters[j].most_common(1)[0][0]
                        template.append(most_common_val)
                        n_fixed += 1
                    else:
                        template.append(None)  # wildcard

                if n_fixed < TEMPLATE_MIN_FIXED:
                    continue

                # Deduplicate sources
                sources = list(set(src for (_, src, _) in members))

                all_templates.append({
                    "template": template,
                    "count": cluster_size,
                    "unique_sources": len(sources),
                    "sources": sorted(sources),
                    "length": L,
                    "fixed_positions": n_fixed,
                    "fixed_ratio": n_fixed / L,
                    "template_str": template_str(template),
                })

        # Sort by (unique_sources desc, count desc) — cross-file patterns first
        all_templates.sort(key=lambda t: (-t["unique_sources"], -t["count"]))

        # Deduplicate: if a shorter template is a sub-pattern of a longer one
        # with equal or fewer sources, keep only the longer one
        return self._deduplicate(all_templates)

    def _deduplicate(self, templates):
        """Remove templates that are strict sub-patterns of longer ones."""
        keep = []
        for i, t in enumerate(templates):
            is_subpattern = False
            for j, other in enumerate(templates):
                if i == j or other["length"] <= t["length"]:
                    continue
                if other["count"] >= t["count"] * 0.8:
                    # Check if t is contained within other
                    if self._is_sub_template(t["template"], other["template"]):
                        is_subpattern = True
                        break
            if not is_subpattern:
                keep.append(t)
        return keep

    @staticmethod
    def _is_sub_template(short, long):
        """Check if short template is contained in long template (matching fixed bytes)."""
        ls, ll = len(short), len(long)
        for start in range(ll - ls + 1):
            match = True
            for k in range(ls):
                if short[k] is not None and long[start + k] is not None:
                    if short[k] != long[start + k]:
                        match = False
                        break
                elif short[k] is not None and long[start + k] is None:
                    match = False
                    break
            if match:
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Positional Byte Entropy Analyzer
# ──────────────────────────────────────────────────────────────────────────────
class PositionalEntropyAnalyzer:
    """
    For a collection of equal-length byte sequences (padded/truncated to
    BOUNDARY_WINDOW), compute per-position entropy.  Positions with low
    entropy are likely opcode bytes; high entropy = operands/immediates.
    """

    def __init__(self, zones, window=BOUNDARY_WINDOW):
        self.window = window
        # Pad/truncate zones to fixed length
        self.zones = []
        for z in zones:
            if len(z) >= window:
                self.zones.append(z[:window])
            else:
                self.zones.append(z + b'\x00' * (window - len(z)))

    def analyze(self):
        if not self.zones:
            return [], []

        n = len(self.zones)
        entropies = []
        top_values = []  # most common value at each position

        for pos in range(self.window):
            ctr = Counter(z[pos] for z in self.zones)
            ent = byte_entropy(ctr, n)
            entropies.append(ent)
            top_val, top_cnt = ctr.most_common(1)[0]
            top_values.append((top_val, top_cnt, n))

        return entropies, top_values

    def report(self, label):
        entropies, top_values = self.analyze()
        if not entropies:
            return f"\n  {label}: no data\n"

        lines = [f"\n{'='*75}",
                 f"  POSITIONAL BYTE ENTROPY — {label}",
                 f"  (low entropy = likely opcode byte, high entropy = likely operand/immediate)",
                 f"{'='*75}"]

        lines.append(f"  {'Pos':>4s}  {'Entropy':>8s}  {'Role':>10s}  "
                     f"{'Top Value':>10s}  {'Freq':>8s}  Visual")
        lines.append(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*30}")

        for pos, (ent, (val, cnt, total)) in enumerate(zip(entropies, top_values)):
            role = "OPCODE?" if ent < 3.0 else ("mixed" if ent < 5.0 else "operand")
            bar = "█" * int(ent * 4)  # visual entropy bar
            pct = f"{100*cnt/total:.0f}%"
            lines.append(f"  {pos:4d}  {ent:8.3f}  {role:>10s}  "
                         f"0x{val:02x} ({val:3d})  {pct:>8s}  {bar}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# ISA Signature Scanner
# ──────────────────────────────────────────────────────────────────────────────
class ISAScanner:
    """Scan all boundary zones for known ISA signatures and tally matches."""

    def __init__(self):
        # sig_name -> list of source_tags
        self.hits = defaultdict(list)
        self.arch_scores = Counter()  # architecture family -> total hits

    def scan_zone(self, data: bytes, zone_label: str, source: str):
        """Scan a byte zone for all known ISA signatures."""
        for (name, desc, pattern, mask) in ISA_SIGNATURES:
            for offset in range(len(data) - len(pattern) + 1):
                if match_signature(data, offset, pattern, mask):
                    self.hits[name].append((source, offset, data[offset:offset+len(pattern)]))
                    # Credit the architecture family
                    arch = name.split()[0]  # "ARM-Thumb", "RL78", etc.
                    self.arch_scores[arch] += 1
                    break  # one match per signature per zone is enough

    def report(self):
        lines = [f"\n{'='*75}",
                 "  ISA HEURISTIC SIGNATURE MATCHES",
                 "  (scanning boundary zones for known prologue/epilogue/call/ret opcodes)",
                 f"{'='*75}"]

        # Architecture scoreboard
        lines.append(f"\n  Architecture scoreboard (total zone hits):")
        for arch, score in self.arch_scores.most_common():
            bar = "█" * min(score, 50)
            lines.append(f"    {arch:<15s}  {score:4d} hits  {bar}")

        # Per-signature detail
        lines.append(f"\n  {'Signature':<30s}  {'Hits':>5s}  Sources")
        lines.append(f"  {'─'*30}  {'─'*5}  {'─'*50}")

        # Sort by hit count descending
        for name, desc, _, _ in ISA_SIGNATURES:
            hit_list = self.hits.get(name, [])
            if not hit_list:
                continue
            # Show unique sources
            unique_sources = sorted(set(src for src, _, _ in hit_list))
            n_show = min(5, len(unique_sources))
            src_str = ", ".join(unique_sources[:n_show])
            if len(unique_sources) > n_show:
                src_str += f", ... (+{len(unique_sources)-n_show} more)"
            lines.append(f"  {name:<30s}  {len(hit_list):5d}  {src_str}")
            lines.append(f"    ↳ {desc}")

        if not any(self.hits.values()):
            lines.append("  No known ISA signatures matched. The ISA may be unusual or")
            lines.append("  the boundary detection may need tuning.")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pattern Miner (extended from v1 with source tracking)
# ──────────────────────────────────────────────────────────────────────────────
class PatternMiner:
    def __init__(self):
        # Exact n-gram counters with full source tracking
        self.prologue_ngrams = Counter()
        self.epilogue_ngrams = Counter()
        self.pre_context_ngrams = Counter()
        self.post_context_ngrams = Counter()

        # Source tracking: pattern_bytes -> list of source_tag strings
        self.prologue_sources = defaultdict(list)
        self.epilogue_sources = defaultdict(list)
        self.pre_context_sources = defaultdict(list)
        self.post_context_sources = defaultdict(list)

        # Single byte / pair histograms with sources
        self.first_bytes = Counter()
        self.last_bytes = Counter()
        self.pre_boundary_bytes = Counter()
        self.post_boundary_bytes = Counter()
        self.first_pairs = Counter()
        self.last_pairs = Counter()
        self.first_pair_sources = defaultdict(list)
        self.last_pair_sources = defaultdict(list)

        # Raw zone storage for wildcard mining and entropy
        self.prologue_zones = []      # (bytes,)
        self.epilogue_zones = []
        self.prologue_labels = []     # parallel source tags
        self.epilogue_labels = []
        self.pre_context_zones = []
        self.post_context_zones = []
        self.pre_context_labels = []
        self.post_context_labels = []

        # Alignment
        self.start_offsets = []
        self.end_offsets = []
        self.region_lengths = []

        self.num_regions = 0
        self.skipped = 0

        # ISA scanner
        self.isa_scanner = ISAScanner()

    def add_region(self, device, filename, bd):
        if bd is None:
            self.skipped += 1
            return
        self.num_regions += 1
        tag = source_tag(device, filename, bd["start"])

        region = bd["full_region"]
        pre = bd["pre_context"]
        post = bd["post_context"]
        prol = bd["prologue_zone"]
        epil = bd["epilogue_zone"]

        # ── Single-byte / pair histograms ──
        if len(region) >= 1:
            self.first_bytes[region[0]] += 1
            self.last_bytes[region[-1]] += 1
        if len(region) >= 2:
            self.first_pairs[region[:2]] += 1
            self.first_pair_sources[region[:2]].append(tag)
            self.last_pairs[region[-2:]] += 1
            self.last_pair_sources[region[-2:]].append(tag)
        if len(pre) >= 1:
            self.pre_boundary_bytes[pre[-1]] += 1
        if len(post) >= 1:
            self.post_boundary_bytes[post[0]] += 1

        # ── Exact n-gram extraction with sources ──
        for ng, _ in self._ngrams(prol):
            self.prologue_ngrams[ng] += 1
            self.prologue_sources[ng].append(tag)
        for ng, _ in self._ngrams(epil):
            self.epilogue_ngrams[ng] += 1
            self.epilogue_sources[ng].append(tag)
        for ng, _ in self._ngrams(pre):
            self.pre_context_ngrams[ng] += 1
            self.pre_context_sources[ng].append(tag)
        for ng, _ in self._ngrams(post):
            self.post_context_ngrams[ng] += 1
            self.post_context_sources[ng].append(tag)

        # ── Store zones ──
        self.prologue_zones.append(prol)
        self.prologue_labels.append(tag)
        self.epilogue_zones.append(epil)
        self.epilogue_labels.append(tag)
        self.pre_context_zones.append(pre)
        self.pre_context_labels.append(tag)
        self.post_context_zones.append(post)
        self.post_context_labels.append(tag)

        # ── Alignment ──
        self.start_offsets.append(bd["start"])
        self.end_offsets.append(bd["end"])
        self.region_lengths.append(bd["end"] - bd["start"])

        # ── ISA scan ──
        self.isa_scanner.scan_zone(prol, "prologue", tag)
        self.isa_scanner.scan_zone(epil, "epilogue", tag)
        self.isa_scanner.scan_zone(pre, "pre_context", tag)
        self.isa_scanner.scan_zone(post, "post_context", tag)

    @staticmethod
    def _ngrams(data, n_min=NGRAM_MIN, n_max=NGRAM_MAX):
        for n in range(n_min, n_max + 1):
            for i in range(len(data) - n + 1):
                yield data[i:i+n], i

    # ── Reporting helpers ──

    def _top_patterns(self, counter, source_map, top_k=TOP_K, min_occ=MIN_OCCURRENCES):
        results = []
        for pat, cnt in counter.most_common(top_k * 5):
            if cnt < min_occ:
                break
            sources = sorted(set(source_map.get(pat, [])))
            results.append({
                "hex": hex_str(pat),
                "count": cnt,
                "length": len(pat),
                "unique_sources": len(sources),
                "sources": sources,
            })
            if len(results) >= top_k:
                break
        return results

    def _byte_histogram(self, counter, label, top_n=20):
        lines = [f"\n{'='*60}", f"  {label}  (top {top_n})", f"{'='*60}"]
        for val, cnt in counter.most_common(top_n):
            if isinstance(val, int):
                lines.append(f"  0x{val:02x}  ({val:3d})  :  {cnt:4d} times  "
                             f"{'#' * min(cnt, 50)}")
            else:
                lines.append(f"  {hex_str(val):14s}  :  {cnt:4d} times  "
                             f"{'#' * min(cnt, 50)}")
        return "\n".join(lines)

    def _pattern_table(self, patterns, label, show_sources=True):
        lines = [f"\n{'='*75}", f"  {label}", f"{'='*75}"]
        lines.append(f"  {'Pattern':<26s}  {'Cnt':>5s}  {'Len':>3s}  "
                     f"{'Srcs':>5s}  Sources")
        lines.append(f"  {'─'*26}  {'─'*5}  {'─'*3}  {'─'*5}  {'─'*50}")

        for p in patterns:
            n_show = min(6, len(p["sources"]))
            src_preview = ", ".join(p["sources"][:n_show])
            if len(p["sources"]) > n_show:
                src_preview += f" (+{len(p['sources'])-n_show} more)"
            lines.append(f"  {p['hex']:<26s}  {p['count']:5d}  {p['length']:3d}  "
                         f"{p['unique_sources']:5d}  {src_preview}")

        return "\n".join(lines)

    def _pair_table(self, counter, source_map, label, top_n=15):
        lines = [f"\n{'='*75}", f"  {label}", f"{'='*75}"]
        lines.append(f"  {'Pair':<8s}  {'Cnt':>5s}  {'Srcs':>5s}  Sources")
        lines.append(f"  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*55}")

        for pair, cnt in counter.most_common(top_n):
            sources = sorted(set(source_map.get(pair, [])))
            n_show = min(6, len(sources))
            src_preview = ", ".join(sources[:n_show])
            if len(sources) > n_show:
                src_preview += f" (+{len(sources)-n_show} more)"
            lines.append(f"  {hex_str(pair):<8s}  {cnt:5d}  {len(sources):5d}  {src_preview}")

        return "\n".join(lines)

    def alignment_report(self):
        lines = [f"\n{'='*60}", "  ALIGNMENT & LENGTH ANALYSIS", f"{'='*60}"]
        for align in ALIGNMENT_CHECK:
            starts_aligned = sum(1 for o in self.start_offsets if o % align == 0)
            ends_aligned = sum(1 for o in self.end_offsets if o % align == 0)
            lens_aligned = sum(1 for l in self.region_lengths if l % align == 0)
            n = len(self.start_offsets) or 1
            lines.append(f"  Alignment to {align}: starts={starts_aligned}/{n} "
                         f"({100*starts_aligned/n:.0f}%)  "
                         f"ends={ends_aligned}/{n} ({100*ends_aligned/n:.0f}%)  "
                         f"lengths={lens_aligned}/{n} ({100*lens_aligned/n:.0f}%)")

        if self.region_lengths:
            lines.append(f"\n  Region length stats:")
            lines.append(f"    min={min(self.region_lengths)}  "
                         f"max={max(self.region_lengths)}  "
                         f"mean={statistics.mean(self.region_lengths):.1f}  "
                         f"median={statistics.median(self.region_lengths):.1f}")
            lines.append(f"\n  Length mod distribution (instruction width hint):")
            for mod in [2, 3, 4]:
                dist = Counter(l % mod for l in self.region_lengths)
                lines.append(f"    mod {mod}: {dict(sorted(dist.items()))}")

        return "\n".join(lines)

    def cross_device_report(self):
        """Exact n-grams (4+ bytes) found in prologue zones across different devices."""
        lines = [f"\n{'='*75}",
                 "  CROSS-DEVICE EXACT PATTERN MATCHES (prologue zones)",
                 f"{'='*75}"]

        # Group sources by device
        pattern_devices = defaultdict(set)  # pattern -> set of device names
        pattern_sources_all = defaultdict(set)

        for pat, srcs in self.prologue_sources.items():
            if len(pat) < 4:
                continue
            for s in srcs:
                dev = s.split("/")[0]
                pattern_devices[pat].add(dev)
                pattern_sources_all[pat].add(s)

        # Keep only patterns seen in 2+ devices
        cross = [(pat, len(devs), devs, pattern_sources_all[pat])
                 for pat, devs in pattern_devices.items() if len(devs) >= 2]
        cross.sort(key=lambda x: (-x[1], -len(x[0])))

        if cross:
            lines.append(f"\n  {'Pattern':<26s}  {'Devs':>5s}  Devices / Sources")
            lines.append(f"  {'─'*26}  {'─'*5}  {'─'*50}")
            for pat, nd, devs, srcs in cross[:30]:
                src_list = sorted(srcs)[:5]
                src_str = ", ".join(src_list)
                if len(srcs) > 5:
                    src_str += f" (+{len(srcs)-5})"
                lines.append(f"  {hex_str(pat):<26s}  {nd:5d}  devs={sorted(devs)}")
                lines.append(f"  {'':26s}        {src_str}")
        else:
            lines.append("  No 4+ byte patterns found across multiple device families.")

        return "\n".join(lines)

    def generate_full_report(self):
        report = []
        sep = "\n" + "=" * 75
        report.append(sep)
        report.append("  FIRMWARE PROLOGUE / EPILOGUE PATTERN MINING REPORT v2")
        report.append(f"  Regions analyzed: {self.num_regions}  |  Skipped: {self.skipped}")
        report.append(f"  Config: context={CONTEXT_BYTES}  window={BOUNDARY_WINDOW}  "
                      f"ngram={NGRAM_MIN}-{NGRAM_MAX}  min_occ={MIN_OCCURRENCES}")
        report.append(sep)

        # ── ISA heuristic signatures (most actionable section) ──
        report.append(self.isa_scanner.report())

        # ── Positional entropy ──
        pro_entropy = PositionalEntropyAnalyzer(self.prologue_zones, BOUNDARY_WINDOW)
        epi_entropy = PositionalEntropyAnalyzer(self.epilogue_zones, BOUNDARY_WINDOW)
        report.append(pro_entropy.report("PROLOGUE ZONE"))
        report.append(epi_entropy.report("EPILOGUE ZONE"))

        # ── Wildcard templates ──
        report.append(f"\n{'='*75}")
        report.append("  WILDCARD TEMPLATE PATTERNS — PROLOGUE ZONES")
        report.append("  (opcode bytes fixed, operand bytes shown as ??)")
        report.append(f"{'='*75}")
        pro_miner = WildcardTemplateMiner(self.prologue_zones, self.prologue_labels)
        pro_templates = pro_miner.mine_templates()
        if pro_templates:
            report.append(f"\n  {'Template':<26s}  {'Cnt':>5s}  {'Srcs':>5s}  "
                         f"{'Fixed':>6s}  Sources")
            report.append(f"  {'─'*26}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*50}")
            for t in pro_templates[:TOP_K]:
                n_show = min(5, len(t["sources"]))
                src_str = ", ".join(t["sources"][:n_show])
                if len(t["sources"]) > n_show:
                    src_str += f" (+{len(t['sources'])-n_show})"
                report.append(f"  {t['template_str']:<26s}  {t['count']:5d}  "
                             f"{t['unique_sources']:5d}  "
                             f"{t['fixed_positions']}/{t['length']}   {src_str}")
        else:
            report.append("  No wildcard templates found meeting threshold.")

        report.append(f"\n{'='*75}")
        report.append("  WILDCARD TEMPLATE PATTERNS — EPILOGUE ZONES")
        report.append(f"{'='*75}")
        epi_miner = WildcardTemplateMiner(self.epilogue_zones, self.epilogue_labels)
        epi_templates = epi_miner.mine_templates()
        if epi_templates:
            report.append(f"\n  {'Template':<26s}  {'Cnt':>5s}  {'Srcs':>5s}  "
                         f"{'Fixed':>6s}  Sources")
            report.append(f"  {'─'*26}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*50}")
            for t in epi_templates[:TOP_K]:
                n_show = min(5, len(t["sources"]))
                src_str = ", ".join(t["sources"][:n_show])
                if len(t["sources"]) > n_show:
                    src_str += f" (+{len(t['sources'])-n_show})"
                report.append(f"  {t['template_str']:<26s}  {t['count']:5d}  "
                             f"{t['unique_sources']:5d}  "
                             f"{t['fixed_positions']}/{t['length']}   {src_str}")
        else:
            report.append("  No wildcard templates found meeting threshold.")

        # ── Pre/post context wildcard templates ──
        report.append(f"\n{'='*75}")
        report.append("  WILDCARD TEMPLATE PATTERNS — PRE-CONTEXT (before region)")
        report.append("  (these may be the EPILOGUE of the previous function)")
        report.append(f"{'='*75}")
        pre_miner = WildcardTemplateMiner(self.pre_context_zones, self.pre_context_labels)
        pre_templates = pre_miner.mine_templates()
        if pre_templates:
            report.append(f"\n  {'Template':<26s}  {'Cnt':>5s}  {'Srcs':>5s}  "
                         f"{'Fixed':>6s}  Sources")
            report.append(f"  {'─'*26}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*50}")
            for t in pre_templates[:25]:
                n_show = min(5, len(t["sources"]))
                src_str = ", ".join(t["sources"][:n_show])
                if len(t["sources"]) > n_show:
                    src_str += f" (+{len(t['sources'])-n_show})"
                report.append(f"  {t['template_str']:<26s}  {t['count']:5d}  "
                             f"{t['unique_sources']:5d}  "
                             f"{t['fixed_positions']}/{t['length']}   {src_str}")

        report.append(f"\n{'='*75}")
        report.append("  WILDCARD TEMPLATE PATTERNS — POST-CONTEXT (after region)")
        report.append("  (these may be the PROLOGUE of the next function)")
        report.append(f"{'='*75}")
        post_miner = WildcardTemplateMiner(self.post_context_zones, self.post_context_labels)
        post_templates = post_miner.mine_templates()
        if post_templates:
            report.append(f"\n  {'Template':<26s}  {'Cnt':>5s}  {'Srcs':>5s}  "
                         f"{'Fixed':>6s}  Sources")
            report.append(f"  {'─'*26}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*50}")
            for t in post_templates[:25]:
                n_show = min(5, len(t["sources"]))
                src_str = ", ".join(t["sources"][:n_show])
                if len(t["sources"]) > n_show:
                    src_str += f" (+{len(t['sources'])-n_show})"
                report.append(f"  {t['template_str']:<26s}  {t['count']:5d}  "
                             f"{t['unique_sources']:5d}  "
                             f"{t['fixed_positions']}/{t['length']}   {src_str}")

        # ── Exact n-gram tables (with sources) ──
        report.append(self._pattern_table(
            self._top_patterns(self.prologue_ngrams, self.prologue_sources),
            "TOP EXACT PROLOGUE N-GRAMS (first 16 bytes of each region)"))

        report.append(self._pattern_table(
            self._top_patterns(self.epilogue_ngrams, self.epilogue_sources),
            "TOP EXACT EPILOGUE N-GRAMS (last 16 bytes of each region)"))

        report.append(self._pattern_table(
            self._top_patterns(self.pre_context_ngrams, self.pre_context_sources),
            "TOP EXACT PRE-CONTEXT N-GRAMS (bytes before region)"))

        report.append(self._pattern_table(
            self._top_patterns(self.post_context_ngrams, self.post_context_sources),
            "TOP EXACT POST-CONTEXT N-GRAMS (bytes after region)"))

        # ── Byte-pair tables with sources ──
        report.append(self._pair_table(
            self.first_pairs, self.first_pair_sources,
            "FIRST 2 BYTES with sources (potential prologue opcode)"))
        report.append(self._pair_table(
            self.last_pairs, self.last_pair_sources,
            "LAST 2 BYTES with sources (potential return opcode)"))

        # ── Single-byte histograms ──
        report.append(self._byte_histogram(self.first_bytes, "FIRST BYTE histogram"))
        report.append(self._byte_histogram(self.last_bytes, "LAST BYTE histogram"))
        report.append(self._byte_histogram(self.pre_boundary_bytes,
                                           "BYTE just BEFORE region start"))
        report.append(self._byte_histogram(self.post_boundary_bytes,
                                           "BYTE just AFTER region end"))

        # ── Alignment ──
        report.append(self.alignment_report())

        # ── Cross-device ──
        report.append(self.cross_device_report())

        # ── Entropy ──
        report.append(self._entropy_summary())

        return "\n".join(report)

    def _entropy_summary(self):
        lines = [f"\n{'='*60}", "  BYTE ENTROPY ANALYSIS", f"{'='*60}"]
        for label, zones in [("Prologue zones", self.prologue_zones),
                             ("Epilogue zones", self.epilogue_zones),
                             ("Pre-context",    self.pre_context_zones),
                             ("Post-context",   self.post_context_zones)]:
            all_bytes = b"".join(zones)
            if not all_bytes:
                continue
            freq = Counter(all_bytes)
            total = len(all_bytes)
            ent = -sum((c / total) * math.log2(c / total) for c in freq.values())
            lines.append(f"  {label}: {total} bytes, Shannon entropy = {ent:.3f} bits/byte")
        lines.append(f"    (random=8.0, English≈4.5, code≈5-7, compressed≈7.5+)")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Per-region hex dump
# ──────────────────────────────────────────────────────────────────────────────
def dump_region_detail(bd, device, filename, row_idx, f_out):
    f_out.write(f"\n{'─'*75}\n")
    f_out.write(f"Region #{row_idx}  |  {device} / {filename}  |  "
                f"offset 0x{bd['start']:06x}–0x{bd['end']:06x}  "
                f"({bd['end'] - bd['start']} bytes)\n")
    f_out.write(f"{'─'*75}\n")

    f_out.write(f"  PRE-CONTEXT  ({len(bd['pre_context'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['pre_context'])}\n")
    f_out.write(f"  >>> PROLOGUE ZONE ({len(bd['prologue_zone'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['prologue_zone'])}\n")

    full = bd['full_region']
    if len(full) > 2 * BOUNDARY_WINDOW:
        mid = len(full) - 2 * BOUNDARY_WINDOW
        f_out.write(f"  ... middle {mid} bytes omitted ...\n")

    f_out.write(f"  <<< EPILOGUE ZONE ({len(bd['epilogue_zone'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['epilogue_zone'])}\n")
    f_out.write(f"  POST-CONTEXT ({len(bd['post_context'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['post_context'])}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global CONTEXT_BYTES, BOUNDARY_WINDOW, NGRAM_MAX, MIN_OCCURRENCES

    parser = argparse.ArgumentParser(
        description="Mine prologue/epilogue patterns (wildcard-aware) from firmware")
    parser.add_argument("--bin-root", required=True,
                        help="Root directory with device subfolders containing .dat files")
    parser.add_argument("--csv", required=True,
                        help="Path to filtered.csv with candidate regions")
    parser.add_argument("--output", default="pattern_report_v2.txt",
                        help="Main report file (default: pattern_report_v2.txt)")
    parser.add_argument("--dump", default="region_details_v2.txt",
                        help="Per-region hex dump file (default: region_details_v2.txt)")
    parser.add_argument("--json", default="patterns_v2.json",
                        help="JSON output (default: patterns_v2.json)")
    parser.add_argument("--context", type=int, default=CONTEXT_BYTES,
                        help=f"Bytes of context around each region (default: {CONTEXT_BYTES})")
    parser.add_argument("--window", type=int, default=BOUNDARY_WINDOW,
                        help=f"Prologue/epilogue zone size (default: {BOUNDARY_WINDOW})")
    parser.add_argument("--ngram-max", type=int, default=NGRAM_MAX,
                        help=f"Max exact n-gram length (default: {NGRAM_MAX})")
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

    # ── Binary cache ──
    file_cache: dict[str, bytes | None] = {}

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

    # ── Process rows ──
    miner = PatternMiner()

    detail_file = open(args.dump, "w", encoding="utf-8")
    detail_file.write("DETAILED REGION HEX DUMPS (v2)\n")
    detail_file.write(f"Context={CONTEXT_BYTES} bytes, Window={BOUNDARY_WINDOW} bytes\n\n")

    for i, row in enumerate(rows):
        device = row["device"].strip()
        side = row.get("side", "").strip()
        try:
            start = int(row["start"])
            end = int(row["end"])
        except (ValueError, KeyError):
            print(f"  WARNING: skipping row {i} — bad start/end", file=sys.stderr)
            continue

        if side == "new":
            filename = row["new_file"].strip()
        elif side == "base":
            filename = row["base_file"].strip()
        else:
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

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")
    print(f"Detail dump written to {args.dump}")

    # ── JSON export (with full source lists) ──
    def json_pattern_list(counter, source_map, top_k=TOP_K, min_occ=MIN_OCCURRENCES):
        result = []
        for pat, cnt in counter.most_common(top_k * 3):
            if cnt < min_occ:
                break
            sources = sorted(set(source_map.get(pat, [])))
            result.append({
                "pattern": hex_str(pat),
                "count": cnt,
                "length": len(pat),
                "unique_sources": len(sources),
                "sources": sources,
            })
            if len(result) >= top_k:
                break
        return result

    json_data = {
        "summary": {
            "regions_analyzed": miner.num_regions,
            "regions_skipped": miner.skipped,
            "config": {
                "context_bytes": CONTEXT_BYTES,
                "boundary_window": BOUNDARY_WINDOW,
                "ngram_range": [NGRAM_MIN, NGRAM_MAX],
                "min_occurrences": MIN_OCCURRENCES,
            }
        },
        "isa_architecture_scores": dict(miner.isa_scanner.arch_scores.most_common()),
        "isa_signature_hits": {
            name: [{"source": s, "offset": o, "matched_bytes": hex_str(b)}
                   for s, o, b in hits]
            for name, hits in miner.isa_scanner.hits.items() if hits
        },
        "prologue_ngrams": json_pattern_list(
            miner.prologue_ngrams, miner.prologue_sources),
        "epilogue_ngrams": json_pattern_list(
            miner.epilogue_ngrams, miner.epilogue_sources),
        "pre_context_ngrams": json_pattern_list(
            miner.pre_context_ngrams, miner.pre_context_sources),
        "post_context_ngrams": json_pattern_list(
            miner.post_context_ngrams, miner.post_context_sources),
        "first_pair_histogram": [
            {"pair": hex_str(p), "count": c,
             "sources": sorted(set(miner.first_pair_sources.get(p, [])))}
            for p, c in miner.first_pairs.most_common(20)
        ],
        "last_pair_histogram": [
            {"pair": hex_str(p), "count": c,
             "sources": sorted(set(miner.last_pair_sources.get(p, [])))}
            for p, c in miner.last_pairs.most_common(20)
        ],
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data written to {args.json}")

    # ── Console summary ──
    print("\n" + "=" * 75)
    print("  QUICK SUMMARY")
    print("=" * 75)

    print("\n  ── ISA Architecture Scoreboard ──")
    for arch, score in miner.isa_scanner.arch_scores.most_common(5):
        print(f"    {arch:<15s}  {score:4d} hits")

    print("\n  ── Top 5 Wildcard Prologue Templates ──")
    pro_miner = WildcardTemplateMiner(miner.prologue_zones, miner.prologue_labels)
    for t in pro_miner.mine_templates()[:5]:
        print(f"    {t['template_str']:<26s}  count={t['count']:3d}  "
              f"sources={t['unique_sources']}  fixed={t['fixed_positions']}/{t['length']}")

    print("\n  ── Top 5 Wildcard Epilogue Templates ──")
    epi_miner = WildcardTemplateMiner(miner.epilogue_zones, miner.epilogue_labels)
    for t in epi_miner.mine_templates()[:5]:
        print(f"    {t['template_str']:<26s}  count={t['count']:3d}  "
              f"sources={t['unique_sources']}  fixed={t['fixed_positions']}/{t['length']}")

    print(f"\n  Full report:  {args.output}")
    print(f"  Hex dumps:    {args.dump}")
    print(f"  JSON data:    {args.json}")
    print()


if __name__ == "__main__":
    main()
