#!/usr/bin/env python3
"""
Prologue / Epilogue Pattern Miner v3 — Split-Group Edition
===========================================================
Bachelor thesis tool for mining subroutine boundaries from binary-diffed
Shimano firmware.

Key improvements over v2:
  • Automatic device-family grouping — splits analysis into Group A (ARM Thumb:
    DUE5000, DUE6100) and Group B (unknown DF/F1 ISA: DUE6001, DUE6002,
    DUE8000, SCE6010), then runs each independently.
  • Refined ISA signatures — adds Renesas V850/RH850, refined RL78 extended
    instructions, and custom signatures for the DF/F1 patterns found in v2.
  • Per-group wildcard mining — with homogeneous ISA per group, the positional
    entropy and wildcard templates should converge properly.
  • Tuned entropy thresholds — lowered for per-group analysis where we expect
    more coherence.

Usage:
    python prologue_epilogue_miner_v3.py \\
        --bin-root <path_to_bin_folder> \\
        --csv filtered.csv

    Optional: --group-a DUE5000,DUE6100  --group-b DUE6001,DUE6002,DUE8000,SCE6010
              to override default groupings.

Output files (per group + combined):
    pattern_report_v3_groupA.txt
    pattern_report_v3_groupB.txt
    pattern_report_v3_combined.txt
    region_details_v3.txt
    patterns_v3.json
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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CONTEXT_BYTES       = 32
BOUNDARY_WINDOW     = 16
NGRAM_MIN           = 2
NGRAM_MAX           = 8
TOP_K               = 40
MIN_OCCURRENCES     = 2     # lowered for per-group (fewer regions)
ALIGNMENT_CHECK     = [2, 4]
WILDCARD_LENGTHS    = [2, 3, 4, 5, 6, 7, 8]
TEMPLATE_MIN_FIXED  = 1
TEMPLATE_MIN_OCC    = 2     # lowered for per-group

# Per-group entropy thresholds (tighter since ISA is homogeneous)
ENTROPY_THRESH_GROUP = 2.5   # within a homogeneous group, byte positions with
                              # entropy below this are "fixed" (likely opcode)
ENTROPY_THRESH_MIXED = 3.0   # for mixed/combined analysis

# Default device groupings based on v2 findings
DEFAULT_GROUP_A = {"DUE5000", "DUE6100"}          # ARM Thumb
DEFAULT_GROUP_B = {"DUE6001", "DUE6002", "DUE8000", "SCE6010"}  # Unknown / DF-F1


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def hex_str(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)

def template_str(template: list) -> str:
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
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

def source_tag(device, filename, offset):
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
# ISA Heuristic Signatures
# ──────────────────────────────────────────────────────────────────────────────
ISA_SIGNATURES = []

def sig(name, desc, pattern_hex, mask_hex=None):
    pat = bytes.fromhex(pattern_hex.replace(" ", ""))
    if mask_hex:
        mask = bytes.fromhex(mask_hex.replace(" ", ""))
        m = [b == 0xFF for b in mask]
    else:
        m = [True] * len(pat)
    ISA_SIGNATURES.append((name, desc, pat, m))

# ── ARM Thumb (Group A focused) ───────────────────────────────────────
sig("Thumb PUSH {..,LR}",
    "push {reglist, lr} — function prologue (B5xx in LE)",
    "00 B5", "00 FF")
sig("Thumb POP {..,PC}",
    "pop {reglist, pc} — function return (BDxx in LE)",
    "00 BD", "00 FF")
sig("Thumb BX LR",
    "bx lr — return via link register",
    "70 47")
sig("Thumb SUB SP,#imm",
    "sub sp, #imm7 — allocate stack frame",
    "80 B0", "80 FF")
sig("Thumb ADD SP,#imm",
    "add sp, #imm7 — deallocate stack frame",
    "00 B0", "80 FF")
sig("Thumb MOV r8-r12,Rm (high)",
    "mov high register — often seen in prologues",
    "00 46", "00 FF")
sig("Thumb BL hi",
    "bl <target> high halfword (F0xx-F7xx in LE)",
    "00 F0", "00 F8")
sig("Thumb BL lo",
    "bl <target> low halfword (F8xx-FFxx in LE)",
    "00 F8", "00 F8")
sig("Thumb NOP",
    "nop (mov r8,r8 = C0 46)",
    "C0 46")
sig("Thumb PUSH {r4,LR}",
    "push {r4, lr} — common callee-save",
    "10 B5")
sig("Thumb PUSH {r4,r5,LR}",
    "push {r4, r5, lr}",
    "30 B5")
sig("Thumb PUSH {r4-r7,LR}",
    "push {r4-r7, lr}",
    "F0 B5")
sig("Thumb POP {r4,PC}",
    "pop {r4, pc} — return restoring r4",
    "10 BD")
sig("Thumb POP {r4,r5,PC}",
    "pop {r4, r5, pc}",
    "30 BD")
sig("Thumb POP {r4-r7,PC}",
    "pop {r4-r7, pc}",
    "F0 BD")

# ── Renesas RL78 (refined, relevant to Group B) ──────────────────────
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
sig("RL78 CALL !addr16",  "call !addr16", "9A 00 00", "FF 00 00")
sig("RL78 CALL $addr20",  "call $addr20", "FC 00 00", "FF 00 00")
sig("RL78 BR !addr16",    "br !addr16",   "9B 00 00", "FF 00 00")
sig("RL78 BR $addr20",    "br $addr20",   "FE 00 00 00", "FF 00 00 00")
sig("RL78 SUBW SP,#imm",  "subw sp, #imm8 — alloc stack",  "DA 00", "FF 00")
sig("RL78 ADDW SP,#imm",  "addw sp, #imm8 — dealloc stack","DA 00", "FF 00")  # same opcode, signed

# ── Renesas V850 / RH850 (LE, variable-length, very common in Japanese electronics) ──
sig("V850 PREPARE",
    "prepare list12, imm5 — save regs + alloc frame (common prologue)",
    "80 07", "80 FF")
sig("V850 DISPOSE",
    "dispose imm5, list12 — restore regs + dealloc frame (common epilogue)",
    "00 06", "60 FF")
sig("V850 JMP [LP]",
    "jmp [lp] — return via link pointer (= ret)",
    "E0 07 E0 01")     # jmp [r31] in V850 encoding
sig("V850 JR disp22",
    "jr disp22 — relative jump",
    "80 07 00 00", "E0 FF 00 00")
sig("V850 JARL disp22,LP",
    "jarl disp22, r31 — call subroutine saving return in LP",
    "E0 07 00 00", "E0 FF 00 00")
sig("V850 ST.W reg,[SP+off]",
    "st.w reg, [sp, offset] — push to stack via store",
    "00 73", "00 FF")
sig("V850 LD.W [SP+off],reg",
    "ld.w [sp, offset], reg — pop from stack via load",
    "00 73", "00 FF")  # different funct7 field
sig("V850 MOV imm5,reg",
    "mov imm5, reg — small immediate move",
    "00 02", "00 FE")
sig("V850 ADD SP,imm",
    "add imm, sp — adjust stack pointer",
    "C0 17", "E0 FF")
sig("V850 NOP",
    "nop (mov r0,r0 in V850)",
    "00 00")

# ── Renesas RX (backup, less likely but included) ─────────────────────
sig("RX PUSHM",  "pushm regs", "6E 00", "FF 00")
sig("RX POPM",   "popm regs",  "6F 00", "FF 00")
sig("RX RTS",    "rts — return from subroutine", "02")
sig("RX BSR.W",  "bsr.w — branch to subroutine", "39 00 00", "FF 00 00")

# ── Renesas SuperH SH-2 ──────────────────────────────────────────────
sig("SH2 MOV.L Rm,@-R15", "push register",      "00 2F", "0F FF")
sig("SH2 MOV.L @R15+,Rm", "pop register",        "00 6F", "0F FF")
sig("SH2 RTS",            "rts — return",          "0B 00")
sig("SH2 BSR disp12",     "bsr — branch to sub",  "00 B0", "00 F0")

# ── Generic padding ──────────────────────────────────────────────────
sig("0xFF padding",  "FF padding between functions", "FF FF FF FF")
sig("0x00 padding",  "00 padding / NOP sled",        "00 00 00 00")


def match_signature(data, offset, pattern, mask):
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
    def __init__(self, zones, labels, min_occ=TEMPLATE_MIN_OCC,
                 entropy_threshold=ENTROPY_THRESH_GROUP):
        self.zones = zones
        self.labels = labels
        self.min_occ = min_occ
        self.entropy_thresh = entropy_threshold

    def mine_templates(self, lengths=WILDCARD_LENGTHS):
        all_templates = []
        for L in lengths:
            windows = []
            for zone, label in zip(self.zones, self.labels):
                for i in range(len(zone) - L + 1):
                    windows.append((zone[i:i+L], label, i))
            if len(windows) < self.min_occ:
                continue

            # Global per-position entropy
            pos_counters = [Counter() for _ in range(L)]
            for (w, _, _) in windows:
                for j in range(L):
                    pos_counters[j][w[j]] += 1
            total_w = len(windows)
            pos_entropy = [byte_entropy(c, total_w) for c in pos_counters]
            anchor_positions = [j for j in range(L) if pos_entropy[j] < self.entropy_thresh]
            if len(anchor_positions) < TEMPLATE_MIN_FIXED:
                continue

            # Group by anchor values
            groups = defaultdict(list)
            for (w, src, off) in windows:
                key = tuple(w[j] for j in anchor_positions)
                groups[key].append((w, src, off))

            for key, members in groups.items():
                if len(members) < self.min_occ:
                    continue

                # Refine per-position entropy within cluster
                cluster_counters = [Counter() for _ in range(L)]
                for (w, _, _) in members:
                    for j in range(L):
                        cluster_counters[j][w[j]] += 1
                cluster_size = len(members)

                template = []
                n_fixed = 0
                for j in range(L):
                    ent = byte_entropy(cluster_counters[j], cluster_size)
                    if ent < 1.0:
                        most_common_val = cluster_counters[j].most_common(1)[0][0]
                        template.append(most_common_val)
                        n_fixed += 1
                    else:
                        template.append(None)

                if n_fixed < TEMPLATE_MIN_FIXED:
                    continue

                sources = sorted(set(src for (_, src, _) in members))
                all_templates.append({
                    "template": template,
                    "count": cluster_size,
                    "unique_sources": len(sources),
                    "sources": sources,
                    "length": L,
                    "fixed_positions": n_fixed,
                    "fixed_ratio": n_fixed / L,
                    "template_str": template_str(template),
                })

        all_templates.sort(key=lambda t: (-t["unique_sources"], -t["count"]))
        return self._deduplicate(all_templates)

    def _deduplicate(self, templates):
        keep = []
        for i, t in enumerate(templates):
            is_sub = False
            for j, other in enumerate(templates):
                if i == j or other["length"] <= t["length"]:
                    continue
                if other["count"] >= t["count"] * 0.8:
                    if self._is_sub_template(t["template"], other["template"]):
                        is_sub = True
                        break
            if not is_sub:
                keep.append(t)
        return keep

    @staticmethod
    def _is_sub_template(short, long):
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
    def __init__(self, zones, window=BOUNDARY_WINDOW):
        self.window = window
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
        top_values = []
        for pos in range(self.window):
            ctr = Counter(z[pos] for z in self.zones)
            ent = byte_entropy(ctr, n)
            entropies.append(ent)
            top_val, top_cnt = ctr.most_common(1)[0]
            # Also collect top-3
            top3 = ctr.most_common(3)
            top_values.append((top_val, top_cnt, n, top3))
        return entropies, top_values

    def report(self, label):
        entropies, top_values = self.analyze()
        if not entropies:
            return f"\n  {label}: no data\n"
        lines = [f"\n{'='*80}",
                 f"  POSITIONAL BYTE ENTROPY — {label}",
                 f"  (low entropy = likely opcode, high = operand/immediate)",
                 f"{'='*80}"]
        lines.append(f"  {'Pos':>4s}  {'Entropy':>8s}  {'Role':>10s}  "
                     f"{'Top1':>12s}  {'Top2':>12s}  {'Top3':>12s}  Visual")
        lines.append(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*20}")

        for pos, (ent, (val, cnt, total, top3)) in enumerate(zip(entropies, top_values)):
            if ent < 2.0:
                role = "★OPCODE★"
            elif ent < 3.5:
                role = "opcode?"
            elif ent < 5.0:
                role = "mixed"
            else:
                role = "operand"

            bar = "█" * int(ent * 3)

            def fmt_top(entry):
                v, c = entry
                return f"0x{v:02x}({100*c/total:2.0f}%)"

            t1 = fmt_top(top3[0]) if len(top3) > 0 else ""
            t2 = fmt_top(top3[1]) if len(top3) > 1 else ""
            t3 = fmt_top(top3[2]) if len(top3) > 2 else ""

            lines.append(f"  {pos:4d}  {ent:8.3f}  {role:>10s}  "
                         f"{t1:>12s}  {t2:>12s}  {t3:>12s}  {bar}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# ISA Signature Scanner
# ──────────────────────────────────────────────────────────────────────────────
class ISAScanner:
    def __init__(self):
        self.hits = defaultdict(list)  # name -> [(source, offset, matched_bytes)]
        self.arch_scores = Counter()

    def scan_zone(self, data, zone_label, source):
        for (name, desc, pattern, mask) in ISA_SIGNATURES:
            for offset in range(len(data) - len(pattern) + 1):
                if match_signature(data, offset, pattern, mask):
                    self.hits[name].append((source, offset, data[offset:offset+len(pattern)]))
                    arch = name.split()[0]
                    self.arch_scores[arch] += 1
                    break

    def report(self):
        lines = [f"\n{'='*80}",
                 "  ISA HEURISTIC SIGNATURE MATCHES",
                 f"{'='*80}"]

        lines.append(f"\n  Architecture scoreboard:")
        for arch, score in self.arch_scores.most_common():
            bar = "█" * min(score, 50)
            lines.append(f"    {arch:<20s}  {score:4d} hits  {bar}")

        lines.append(f"\n  {'Signature':<35s}  {'Hits':>5s}  Sources")
        lines.append(f"  {'─'*35}  {'─'*5}  {'─'*50}")

        for name, desc, _, _ in ISA_SIGNATURES:
            hit_list = self.hits.get(name, [])
            if not hit_list:
                continue
            unique_sources = sorted(set(src for src, _, _ in hit_list))
            n_show = min(5, len(unique_sources))
            src_str = ", ".join(unique_sources[:n_show])
            if len(unique_sources) > n_show:
                src_str += f", ... (+{len(unique_sources)-n_show} more)"
            lines.append(f"  {name:<35s}  {len(hit_list):5d}  {src_str}")
            lines.append(f"    ↳ {desc}")

        if not any(self.hits.values()):
            lines.append("  No ISA signatures matched.")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern Miner
# ──────────────────────────────────────────────────────────────────────────────
class PatternMiner:
    def __init__(self, group_name="ALL", entropy_threshold=ENTROPY_THRESH_GROUP):
        self.group_name = group_name
        self.entropy_threshold = entropy_threshold

        self.prologue_ngrams = Counter()
        self.epilogue_ngrams = Counter()
        self.pre_context_ngrams = Counter()
        self.post_context_ngrams = Counter()

        self.prologue_sources = defaultdict(list)
        self.epilogue_sources = defaultdict(list)
        self.pre_context_sources = defaultdict(list)
        self.post_context_sources = defaultdict(list)

        self.first_bytes = Counter()
        self.last_bytes = Counter()
        self.pre_boundary_bytes = Counter()
        self.post_boundary_bytes = Counter()
        self.first_pairs = Counter()
        self.last_pairs = Counter()
        self.first_pair_sources = defaultdict(list)
        self.last_pair_sources = defaultdict(list)

        self.prologue_zones = []
        self.epilogue_zones = []
        self.prologue_labels = []
        self.epilogue_labels = []
        self.pre_context_zones = []
        self.post_context_zones = []
        self.pre_context_labels = []
        self.post_context_labels = []

        self.start_offsets = []
        self.end_offsets = []
        self.region_lengths = []

        self.num_regions = 0
        self.skipped = 0

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

        self.prologue_zones.append(prol)
        self.prologue_labels.append(tag)
        self.epilogue_zones.append(epil)
        self.epilogue_labels.append(tag)
        self.pre_context_zones.append(pre)
        self.pre_context_labels.append(tag)
        self.post_context_zones.append(post)
        self.post_context_labels.append(tag)

        self.start_offsets.append(bd["start"])
        self.end_offsets.append(bd["end"])
        self.region_lengths.append(bd["end"] - bd["start"])

        self.isa_scanner.scan_zone(prol, "prologue", tag)
        self.isa_scanner.scan_zone(epil, "epilogue", tag)
        self.isa_scanner.scan_zone(pre, "pre_context", tag)
        self.isa_scanner.scan_zone(post, "post_context", tag)

    @staticmethod
    def _ngrams(data, n_min=NGRAM_MIN, n_max=NGRAM_MAX):
        for n in range(n_min, n_max + 1):
            for i in range(len(data) - n + 1):
                yield data[i:i+n], i

    def _top_patterns(self, counter, source_map, top_k=TOP_K, min_occ=MIN_OCCURRENCES):
        results = []
        for pat, cnt in counter.most_common(top_k * 5):
            if cnt < min_occ:
                break
            sources = sorted(set(source_map.get(pat, [])))
            results.append({
                "hex": hex_str(pat), "count": cnt, "length": len(pat),
                "unique_sources": len(sources), "sources": sources,
            })
            if len(results) >= top_k:
                break
        return results

    def _byte_histogram(self, counter, label, top_n=20):
        lines = [f"\n{'='*65}", f"  {label}  (top {top_n})", f"{'='*65}"]
        for val, cnt in counter.most_common(top_n):
            if isinstance(val, int):
                pct = f"({100*cnt/max(self.num_regions,1):.0f}%)"
                lines.append(f"  0x{val:02x}  ({val:3d})  :  {cnt:4d} times {pct:>6s}  "
                             f"{'█' * min(cnt, 40)}")
            else:
                lines.append(f"  {hex_str(val):14s}  :  {cnt:4d} times  "
                             f"{'█' * min(cnt, 40)}")
        return "\n".join(lines)

    def _pattern_table(self, patterns, label):
        lines = [f"\n{'='*80}", f"  {label}", f"{'='*80}"]
        lines.append(f"  {'Pattern':<26s}  {'Cnt':>5s}  {'Len':>3s}  "
                     f"{'Srcs':>5s}  Sources")
        lines.append(f"  {'─'*26}  {'─'*5}  {'─'*3}  {'─'*5}  {'─'*55}")
        for p in patterns:
            n_show = min(6, len(p["sources"]))
            src_preview = ", ".join(p["sources"][:n_show])
            if len(p["sources"]) > n_show:
                src_preview += f" (+{len(p['sources'])-n_show} more)"
            lines.append(f"  {p['hex']:<26s}  {p['count']:5d}  {p['length']:3d}  "
                         f"{p['unique_sources']:5d}  {src_preview}")
        return "\n".join(lines)

    def _pair_table(self, counter, source_map, label, top_n=15):
        lines = [f"\n{'='*80}", f"  {label}", f"{'='*80}"]
        lines.append(f"  {'Pair':<8s}  {'Cnt':>5s}  {'Srcs':>5s}  Sources")
        lines.append(f"  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*55}")
        for pair, cnt in counter.most_common(top_n):
            sources = sorted(set(source_map.get(pair, [])))
            n_show = min(6, len(sources))
            src_preview = ", ".join(sources[:n_show])
            if len(sources) > n_show:
                src_preview += f" (+{len(sources)-n_show})"
            lines.append(f"  {hex_str(pair):<8s}  {cnt:5d}  {len(sources):5d}  {src_preview}")
        return "\n".join(lines)

    def _wildcard_section(self, zones, labels, section_label):
        lines = [f"\n{'='*80}",
                 f"  WILDCARD TEMPLATES — {section_label}",
                 f"  (opcode bytes fixed, operand bytes = ??)",
                 f"{'='*80}"]
        miner = WildcardTemplateMiner(zones, labels,
                                       min_occ=max(2, self.num_regions // 8),
                                       entropy_threshold=self.entropy_threshold)
        templates = miner.mine_templates()
        if templates:
            lines.append(f"\n  {'Template':<30s}  {'Cnt':>5s}  {'Srcs':>5s}  "
                         f"{'Fixed':>6s}  Sources")
            lines.append(f"  {'─'*30}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*45}")
            for t in templates[:TOP_K]:
                n_show = min(5, len(t["sources"]))
                src_str = ", ".join(t["sources"][:n_show])
                if len(t["sources"]) > n_show:
                    src_str += f" (+{len(t['sources'])-n_show})"
                lines.append(f"  {t['template_str']:<30s}  {t['count']:5d}  "
                             f"{t['unique_sources']:5d}  "
                             f"{t['fixed_positions']}/{t['length']}   {src_str}")
        else:
            lines.append("  No wildcard templates found meeting threshold.")
        return "\n".join(lines), templates

    def alignment_report(self):
        lines = [f"\n{'='*65}", "  ALIGNMENT & LENGTH ANALYSIS", f"{'='*65}"]
        for align in ALIGNMENT_CHECK:
            n = max(len(self.start_offsets), 1)
            sa = sum(1 for o in self.start_offsets if o % align == 0)
            ea = sum(1 for o in self.end_offsets if o % align == 0)
            la = sum(1 for l in self.region_lengths if l % align == 0)
            lines.append(f"  Align {align}: starts={sa}/{n} ({100*sa/n:.0f}%)  "
                         f"ends={ea}/{n} ({100*ea/n:.0f}%)  "
                         f"lengths={la}/{n} ({100*la/n:.0f}%)")
        if self.region_lengths:
            lines.append(f"\n  Length: min={min(self.region_lengths)}  "
                         f"max={max(self.region_lengths)}  "
                         f"mean={statistics.mean(self.region_lengths):.1f}  "
                         f"median={statistics.median(self.region_lengths):.1f}")
            for mod in [2, 3, 4]:
                dist = Counter(l % mod for l in self.region_lengths)
                lines.append(f"    mod {mod}: {dict(sorted(dist.items()))}")
        return "\n".join(lines)

    def _entropy_summary(self):
        lines = [f"\n{'='*65}", "  BYTE ENTROPY SUMMARY", f"{'='*65}"]
        for lbl, zones in [("Prologue zones", self.prologue_zones),
                           ("Epilogue zones", self.epilogue_zones),
                           ("Pre-context",    self.pre_context_zones),
                           ("Post-context",   self.post_context_zones)]:
            ab = b"".join(zones)
            if not ab:
                continue
            freq = Counter(ab)
            total = len(ab)
            ent = -sum((c/total)*math.log2(c/total) for c in freq.values())
            lines.append(f"  {lbl}: {total} bytes, entropy = {ent:.3f} bits/byte")
        lines.append(f"    (random=8.0, text≈4.5, code≈5-7, compressed≈7.5+)")
        return "\n".join(lines)

    def generate_report(self):
        report = []
        sep = "\n" + "=" * 80
        report.append(sep)
        report.append(f"  PATTERN MINING REPORT — {self.group_name}")
        report.append(f"  Regions: {self.num_regions}  |  Skipped: {self.skipped}")
        report.append(sep)

        # ISA signatures
        report.append(self.isa_scanner.report())

        # Positional entropy
        pro_ea = PositionalEntropyAnalyzer(self.prologue_zones, BOUNDARY_WINDOW)
        epi_ea = PositionalEntropyAnalyzer(self.epilogue_zones, BOUNDARY_WINDOW)
        pre_ea = PositionalEntropyAnalyzer(self.pre_context_zones, BOUNDARY_WINDOW)
        post_ea = PositionalEntropyAnalyzer(self.post_context_zones, BOUNDARY_WINDOW)
        report.append(pro_ea.report(f"PROLOGUE ZONE ({self.group_name})"))
        report.append(epi_ea.report(f"EPILOGUE ZONE ({self.group_name})"))
        report.append(pre_ea.report(f"PRE-CONTEXT ({self.group_name}) — may contain previous func epilogue"))
        report.append(post_ea.report(f"POST-CONTEXT ({self.group_name}) — may contain next func prologue"))

        # Wildcard templates for all 4 zones
        wt_text, wt_pro = self._wildcard_section(
            self.prologue_zones, self.prologue_labels, f"PROLOGUE ({self.group_name})")
        report.append(wt_text)

        wt_text, wt_epi = self._wildcard_section(
            self.epilogue_zones, self.epilogue_labels, f"EPILOGUE ({self.group_name})")
        report.append(wt_text)

        wt_text, _ = self._wildcard_section(
            self.pre_context_zones, self.pre_context_labels,
            f"PRE-CONTEXT ({self.group_name}) — previous func epilogue?")
        report.append(wt_text)

        wt_text, _ = self._wildcard_section(
            self.post_context_zones, self.post_context_labels,
            f"POST-CONTEXT ({self.group_name}) — next func prologue?")
        report.append(wt_text)

        # Exact n-grams
        report.append(self._pattern_table(
            self._top_patterns(self.prologue_ngrams, self.prologue_sources),
            f"EXACT PROLOGUE N-GRAMS ({self.group_name})"))
        report.append(self._pattern_table(
            self._top_patterns(self.epilogue_ngrams, self.epilogue_sources),
            f"EXACT EPILOGUE N-GRAMS ({self.group_name})"))
        report.append(self._pattern_table(
            self._top_patterns(self.pre_context_ngrams, self.pre_context_sources),
            f"EXACT PRE-CONTEXT N-GRAMS ({self.group_name})"))
        report.append(self._pattern_table(
            self._top_patterns(self.post_context_ngrams, self.post_context_sources),
            f"EXACT POST-CONTEXT N-GRAMS ({self.group_name})"))

        # Byte pair tables
        report.append(self._pair_table(
            self.first_pairs, self.first_pair_sources,
            f"FIRST 2 BYTES ({self.group_name}) — prologue opcode?"))
        report.append(self._pair_table(
            self.last_pairs, self.last_pair_sources,
            f"LAST 2 BYTES ({self.group_name}) — return opcode?"))

        # Single-byte histograms
        report.append(self._byte_histogram(self.first_bytes,
                                           f"FIRST BYTE ({self.group_name})"))
        report.append(self._byte_histogram(self.last_bytes,
                                           f"LAST BYTE ({self.group_name})"))
        report.append(self._byte_histogram(self.pre_boundary_bytes,
                                           f"BYTE BEFORE region ({self.group_name})"))
        report.append(self._byte_histogram(self.post_boundary_bytes,
                                           f"BYTE AFTER region ({self.group_name})"))

        # Alignment
        report.append(self.alignment_report())

        # Entropy
        report.append(self._entropy_summary())

        return "\n".join(report), wt_pro, wt_epi


# ──────────────────────────────────────────────────────────────────────────────
# Per-region hex dump
# ──────────────────────────────────────────────────────────────────────────────
def dump_region_detail(bd, device, filename, row_idx, group, f_out):
    f_out.write(f"\n{'─'*80}\n")
    f_out.write(f"Region #{row_idx}  |  {group}  |  {device}/{filename}  |  "
                f"0x{bd['start']:06x}–0x{bd['end']:06x}  "
                f"({bd['end'] - bd['start']} bytes)\n")
    f_out.write(f"{'─'*80}\n")
    f_out.write(f"  PRE-CONTEXT  ({len(bd['pre_context'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['pre_context'])}\n")
    f_out.write(f"  >>> PROLOGUE ({len(bd['prologue_zone'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['prologue_zone'])}\n")
    full = bd['full_region']
    if len(full) > 2 * BOUNDARY_WINDOW:
        f_out.write(f"  ... middle {len(full) - 2*BOUNDARY_WINDOW} bytes omitted ...\n")
    f_out.write(f"  <<< EPILOGUE ({len(bd['epilogue_zone'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['epilogue_zone'])}\n")
    f_out.write(f"  POST-CONTEXT ({len(bd['post_context'])} bytes):\n")
    f_out.write(f"    {hex_str(bd['post_context'])}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global CONTEXT_BYTES, BOUNDARY_WINDOW, NGRAM_MAX, MIN_OCCURRENCES
    parser = argparse.ArgumentParser(
        description="Mine prologue/epilogue patterns v3 — split by device group")
    parser.add_argument("--bin-root", required=True,
                        help="Root dir with device subfolders containing .dat files")
    parser.add_argument("--csv", required=True,
                        help="Path to filtered.csv")
    parser.add_argument("--output-prefix", default="pattern_report_v3",
                        help="Prefix for output files (default: pattern_report_v3)")
    parser.add_argument("--dump", default="region_details_v3.txt",
                        help="Per-region hex dump file")
    parser.add_argument("--json", default="patterns_v3.json",
                        help="JSON output file")
    parser.add_argument("--group-a", default=None,
                        help="Comma-separated device names for Group A (ARM Thumb)")
    parser.add_argument("--group-b", default=None,
                        help="Comma-separated device names for Group B (unknown ISA)")
    parser.add_argument("--context", type=int, default=CONTEXT_BYTES)
    parser.add_argument("--window", type=int, default=BOUNDARY_WINDOW)
    parser.add_argument("--ngram-max", type=int, default=NGRAM_MAX)
    parser.add_argument("--min-occ", type=int, default=MIN_OCCURRENCES)
    args = parser.parse_args()

    CONTEXT_BYTES = args.context
    BOUNDARY_WINDOW = args.window
    NGRAM_MAX = args.ngram_max
    MIN_OCCURRENCES = args.min_occ

    group_a = set(args.group_a.split(",")) if args.group_a else DEFAULT_GROUP_A
    group_b = set(args.group_b.split(",")) if args.group_b else DEFAULT_GROUP_B

    bin_root = args.bin_root
    if not os.path.isdir(bin_root):
        print(f"ERROR: bin-root '{bin_root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Load CSV
    rows = []
    with open(args.csv, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"Loaded {len(rows)} candidate regions from {args.csv}")
    print(f"Group A (ARM Thumb): {sorted(group_a)}")
    print(f"Group B (unknown):   {sorted(group_b)}")

    # Binary cache
    file_cache = {}
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

    # Create miners
    miner_a = PatternMiner("Group A — ARM Thumb (DUE5000, DUE6100)",
                           entropy_threshold=ENTROPY_THRESH_GROUP)
    miner_b = PatternMiner("Group B — Unknown ISA (DUE6001, DUE6002, DUE8000, SCE6010)",
                           entropy_threshold=ENTROPY_THRESH_GROUP)
    miner_all = PatternMiner("ALL DEVICES (combined)",
                             entropy_threshold=ENTROPY_THRESH_MIXED)

    detail_file = open(args.dump, "w", encoding="utf-8")
    detail_file.write("DETAILED REGION HEX DUMPS (v3 — split by group)\n\n")

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
            miner_a.skipped += 1
            miner_b.skipped += 1
            miner_all.skipped += 1
            continue

        bd = extract_boundary_bytes(data, start, end, CONTEXT_BYTES)

        # Route to appropriate group
        if device in group_a:
            miner_a.add_region(device, filename, bd)
            group_label = "Group A"
        elif device in group_b:
            miner_b.add_region(device, filename, bd)
            group_label = "Group B"
        else:
            group_label = "Ungrouped"
            print(f"  WARNING: device '{device}' not in any group, skipping group assignment")

        miner_all.add_region(device, filename, bd)

        if bd is not None:
            dump_region_detail(bd, device, filename, i, group_label, detail_file)

    detail_file.close()

    print(f"\nGroup A: {miner_a.num_regions} regions")
    print(f"Group B: {miner_b.num_regions} regions")
    print(f"Total:   {miner_all.num_regions} regions")

    # Generate reports
    reports = {}
    json_data = {"groups": {}}

    for label, miner, suffix in [
        ("Group A", miner_a, "groupA"),
        ("Group B", miner_b, "groupB"),
        ("Combined", miner_all, "combined"),
    ]:
        if miner.num_regions == 0:
            print(f"  {label}: no regions, skipping report.")
            continue

        report_text, wt_pro, wt_epi = miner.generate_report()
        outfile = f"{args.output_prefix}_{suffix}.txt"
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"  {label} report → {outfile}")
        reports[suffix] = report_text

        # JSON
        json_data["groups"][suffix] = {
            "name": miner.group_name,
            "regions": miner.num_regions,
            "isa_scores": dict(miner.isa_scanner.arch_scores.most_common()),
            "wildcard_prologue_templates": [
                {"template": t["template_str"], "count": t["count"],
                 "sources": t["sources"], "fixed": f"{t['fixed_positions']}/{t['length']}"}
                for t in (wt_pro or [])[:20]
            ],
            "wildcard_epilogue_templates": [
                {"template": t["template_str"], "count": t["count"],
                 "sources": t["sources"], "fixed": f"{t['fixed_positions']}/{t['length']}"}
                for t in (wt_epi or [])[:20]
            ],
            "top_prologue_ngrams": [
                {"pattern": p["hex"], "count": p["count"],
                 "sources": p["sources"]}
                for p in miner._top_patterns(
                    miner.prologue_ngrams, miner.prologue_sources, top_k=20)
            ],
            "top_epilogue_ngrams": [
                {"pattern": p["hex"], "count": p["count"],
                 "sources": p["sources"]}
                for p in miner._top_patterns(
                    miner.epilogue_ngrams, miner.epilogue_sources, top_k=20)
            ],
        }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON → {args.json}")
    print(f"  Hex dumps → {args.dump}")

    # Console summary
    print("\n" + "=" * 80)
    print("  QUICK SUMMARY")
    print("=" * 80)

    for label, miner in [("Group A (ARM Thumb)", miner_a),
                          ("Group B (Unknown)", miner_b)]:
        if miner.num_regions == 0:
            continue
        print(f"\n  ── {label} ({miner.num_regions} regions) ──")

        print(f"    ISA scoreboard:")
        for arch, score in miner.isa_scanner.arch_scores.most_common(5):
            print(f"      {arch:<20s}  {score:4d} hits")

        wt = WildcardTemplateMiner(
            miner.prologue_zones, miner.prologue_labels,
            min_occ=max(2, miner.num_regions // 8),
            entropy_threshold=miner.entropy_threshold)
        print(f"    Top wildcard PROLOGUE templates:")
        for t in wt.mine_templates()[:5]:
            print(f"      {t['template_str']:<30s}  cnt={t['count']:3d}  "
                  f"srcs={t['unique_sources']}  fixed={t['fixed_positions']}/{t['length']}")

        wt2 = WildcardTemplateMiner(
            miner.epilogue_zones, miner.epilogue_labels,
            min_occ=max(2, miner.num_regions // 8),
            entropy_threshold=miner.entropy_threshold)
        print(f"    Top wildcard EPILOGUE templates:")
        for t in wt2.mine_templates()[:5]:
            print(f"      {t['template_str']:<30s}  cnt={t['count']:3d}  "
                  f"srcs={t['unique_sources']}  fixed={t['fixed_positions']}/{t['length']}")

    print()


if __name__ == "__main__":
    main()
