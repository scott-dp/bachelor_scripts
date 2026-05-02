# periodicity_autocorr.py
# Analyses byte-stream periodicity as a signal for instruction width.
#
# Two measurements, plotted for all input files on shared axes:
#   1) Autocorrelation at lags 1..64 (Pearson correlation between the
#      byte series and lagged versions of itself). A peak at even lags
#      is consistent with a 2-byte aligned encoding such as ARM Thumb.
#   2) Phase separation score for assumed widths k=2..8. The byte stream
#      is split into k groups by index mod k. The average Jensen-Shannon
#      divergence across all group-histogram pairs is the score for that k.
#
# Usage:
#   py periodicity_autocorr.py --files DUE5000-D.4.1.0.dat DUE6001-D.3.4.7.dat
#   py periodicity_autocorr.py --files *.dat --header-skip 0

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def read_bytes(path, header_skip):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return raw[header_skip:]


def autocorr_byte_series(x: np.ndarray, max_lag=64):
    # Pearson correlation between the byte series and lagged versions of itself.
    if len(x) < max_lag + 2:
        max_lag = max(1, len(x) - 2)
    xf = x.astype(np.float64)
    out = np.zeros(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        a = xf[:len(xf) - lag]
        b = xf[lag:]
        r = np.corrcoef(a, b)[0, 1]
        out[lag - 1] = r if not np.isnan(r) else 0.0
    return out


def hist256_np(arr):
    h = np.bincount(arr, minlength=256).astype(np.float64)
    s = h.sum()
    if s > 0:
        h /= s
    return h


def js_divergence(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))


def phase_separation_score(data: np.ndarray, k: int):
    # Split byte stream into k groups by index mod k.
    # Score = mean JS divergence across all group-histogram pairs.
    hists = [hist256_np(data[p::k]) for p in range(k)]
    pairs = [js_divergence(hists[i], hists[j])
             for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(pairs)) if pairs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="One or more firmware blobs (.dat/.bin)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Legend labels for each file (default: filename stem)")
    ap.add_argument("--max-lag", type=int, default=64)
    ap.add_argument("--header-skip", type=int, default=6000,
                    help="Bytes to skip at file start (default: 6000)")
    ap.add_argument("--save-prefix", default="periodicity")
    args = ap.parse_args()

    # Default labels = filename without extension
    labels = args.labels if args.labels else \
             [os.path.splitext(os.path.basename(f))[0] for f in args.files]

    if len(labels) != len(args.files):
        ap.error("--labels must have the same number of entries as --files")

    ks = np.arange(2, 9)

    fig_ac, ax_ac = plt.subplots(figsize=(12, 5))
    fig_ps, ax_ps = plt.subplots(figsize=(8, 4))

    for path, label in zip(args.files, labels):
        data = read_bytes(path, args.header_skip)
        print(f"\n{label}")
        print(f"  Bytes analysed: {len(data)}")

        # Autocorrelation
        ac   = autocorr_byte_series(data, max_lag=args.max_lag)
        lags = np.arange(1, len(ac) + 1)
        ax_ac.plot(lags, ac, label=label)

        # Phase separation
        phase_scores = np.array([phase_separation_score(data, int(k)) for k in ks])
        print("  Phase separation scores:")
        for k, s in zip(ks, phase_scores):
            print(f"    k={k}: {s:.6f}")
        best_k = int(ks[np.argmax(phase_scores)])
        print(f"  Best k: {best_k}")
        ax_ps.plot(ks, phase_scores, marker="o", label=label)

    # Autocorrelation plot styling
    ax_ac.set_xticks(np.arange(0, args.max_lag + 1, 4))
    ax_ac.set_title("Byte Autocorrelation vs Lag")
    ax_ac.set_xlabel("Lag (bytes)")
    ax_ac.set_ylabel("Autocorrelation")
    ax_ac.legend()
    ax_ac.grid(True, alpha=0.3)
    fig_ac.tight_layout()
    fig_ac.savefig(f"{args.save_prefix}_autocorr.png", dpi=150)

    # Phase separation plot styling
    ax_ps.set_xticks(ks)
    ax_ps.set_title("Phase Separation Score by Assumed Instruction Width")
    ax_ps.set_xlabel("Assumed width k (bytes)")
    ax_ps.set_ylabel("Mean JS divergence between phase histograms")
    ax_ps.legend()
    ax_ps.grid(True, alpha=0.3)
    fig_ps.tight_layout()
    fig_ps.savefig(f"{args.save_prefix}_phase_scores.png", dpi=150)

    print(f"\nSaved:")
    print(f"  {args.save_prefix}_autocorr.png")
    print(f"  {args.save_prefix}_phase_scores.png")


if __name__ == "__main__":
    main()
