#!/usr/bin/env python3
# Phase 6 (wave-family spectrum): frequency content of the 6-wave test
# battery p6_analysis_dataset_comparison.py evaluates every model against
# (TEST_WAVES there: gaussian, chirp, shock, filtered_random, sawtooth,
# sinusoid -- see that file's own header for what each family is). That
# script only reports model performance per family; this one characterizes
# the waves themselves, independent of any trained model -- no torch, no FD
# solve, just the raw driving signal from physics/waves.py.
#
# Reuses TEST_WAVES/build_bcspec/repeat_seeds/WAVE_LABELS/N_REPEATS from
# p6_analysis_dataset_comparison.py rather than redefining them, so this
# stays locked to the exact same families/seeds/parameter ranges as the
# rest of the p6 report.
#
# For each family: N_REPEATS=20 parameter draws (same reproducible child
# seeds as the rest of p6), each Hann-windowed and FFT'd (same convention as
# evaluate/metrics.py's compute_spectrum), then the 20 power spectra are
# averaged into one representative spectrum per family -- not a single
# lucky/unlucky draw. That averaged spectrum is reduced to one number, its
# power-weighted mean frequency (spectral centroid, in Hz), so all 6
# families can sit in one table.
#
# Usage: python analysis/p6_analysis_wave_spectrum.py
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from _common import render_latex_table, render_readable_table  # noqa: E402
from p6_analysis_dataset_comparison import (  # noqa: E402
    TEST_WAVES, WAVE_LABELS, LATEX_WAVE_LABELS, N_REPEATS, build_bcspec, repeat_seeds,
)
from beamsurrogate.config import load_config  # noqa: E402
from beamsurrogate.physics.waves import BC_WAVEFORMS  # noqa: E402

COMP_DIR = REPO_ROOT / "dataset_comparison" / "runs" / "p6_diag_dataset_comparison"
CFG_PATH = REPO_ROOT / "dataset_comparison" / "runs" / "p6_dataset_simple" / "config.yaml"

COLORS = {"gaussian": "#2a78d6", "chirp": "#eb6834", "shock": "#1baf7a",
          "filtered_random": "#d6272a", "sawtooth": "#9467bd", "sinusoid": "#8c6d31"}


def family_signal(family: str, seed: int, cfg, t: np.ndarray) -> np.ndarray:
    # bc_right only -- the "R" excitation's driven end is this family's own
    # waveform; left/right/anti-phase (see build_bcspec) doesn't change a
    # family's own frequency content, so excitation="R" is enough here.
    _, (_, _, params) = build_bcspec(family, seed, "R", cfg)
    _, value_fn = BC_WAVEFORMS[family]
    # Elementwise, not value_fn(params, t) vectorized over the array:
    # sawtooth_value/triangular_value/square_value branch on a scalar `if`
    # inside _pulse_window and raise on array input.
    return np.array([value_fn(params, float(ti)) for ti in t])


# Zero-padded FFT length. The physical signal is only 501 samples (5s @
# dt=0.01s) -> un-padded bin spacing is 1/5.01 ~= 0.1996 Hz, coarser than
# OMEGA_MIN/OMEGA_MAX converted to Hz (~0.008-0.08 Hz) -- i.e. every
# sinusoid/chirp draw's own fundamental frequency would fall INSIDE the
# first bin, indistinguishable from DC. Zero-padding to 32768 samples gives
# a 0.00305 Hz grid so the centroid integral (and the plot) actually resolve
# that range. This does NOT add information the 5s window doesn't have --
# the true (Rayleigh) resolving power is still set by the 5s duration, so
# it can't separate two nearby true frequencies -- it only interpolates the
# existing spectrum finely enough for an accurate weighted-average frequency
# and a smooth curve, instead of a handful of coarse, misleading steps.
N_FFT = 32768


def family_power_spectra(family: str, seed: int, cfg, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # One Hann-windowed power spectrum per repeat draw (rows), same
    # convention as evaluate/metrics.py's compute_spectrum. Averaging these
    # rows gives the family's representative spectrum; centroiding each row
    # individually gives the per-draw spread used as a robustness check.
    window = np.hanning(len(t))
    freqs = np.fft.rfftfreq(N_FFT, d=cfg.dt)
    powers = np.array([
        np.abs(np.fft.rfft(family_signal(family, child_seed, cfg, t) * window, n=N_FFT)) ** 2
        for child_seed in repeat_seeds(seed, N_REPEATS)
    ])
    return freqs, powers


def spectral_centroid(freqs: np.ndarray, power: np.ndarray) -> float:
    total = power.sum()
    return float((freqs * power).sum() / total) if total > 0 else 0.0


def main():
    figures_dir = COMP_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CFG_PATH)
    t = np.arange(cfg.Nt + 1) * cfg.dt

    fig, ax = plt.subplots(figsize=(8, 5))
    rows_tex, rows_readable, summary_lines = [], [], []

    for family, seed, _excitations in TEST_WAVES:
        freqs, powers = family_power_spectra(family, seed, cfg, t)
        power = powers.mean(axis=0)
        centroid = spectral_centroid(freqs, power)

        # Per-repeat centroids -- a robustness check on the averaged
        # spectrum's own centroid, not the primary number, but worth
        # surfacing in case one family's draws disagree a lot.
        centroid_std = float(np.std([spectral_centroid(freqs, p) for p in powers]))

        color = COLORS[family]
        ax.plot(freqs, power, color=color, label=WAVE_LABELS[family])
        ax.axvline(centroid, color=color, ls="--", lw=1.0)

        rows_tex.append([LATEX_WAVE_LABELS[family], f"{centroid:.4f}"])
        rows_readable.append([WAVE_LABELS[family], f"{centroid:.4f}"])
        summary_lines.append(f"{WAVE_LABELS[family]:22s} f_centroid = {centroid:.4f} Hz "
                              f"(std across {N_REPEATS} draws: {centroid_std:.4f} Hz)")

    # Log-x: every family's centroid sits under 0.3 Hz (OMEGA_MAX=0.5 rad/s
    # is ~0.08 Hz) while the Nyquist edge is 50 Hz -- a linear axis crushes
    # all the actually-informative low-frequency structure against x=0.
    # Capped at 5 Hz: beyond that every family has decayed 15+ orders of
    # magnitude below its own peak, i.e. FFT/float64 noise floor, not signal.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(freqs[1], 5.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (averaged over 20 draws, a.u.)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = figures_dir / "wave_spectrum_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    table_tex = render_latex_table(
        caption="Spectral centroid (power-weighted mean frequency) of each wave family in the "
                "phase-6 test battery, averaged over 20 random parameter draws per family.",
        label="tab:wave_spectrum_centroid",
        col_spec="lc",
        header_cells=["Wave family", "$f_{\\mathrm{centroid}}$ (Hz)"],
        rows=rows_tex,
    )
    (COMP_DIR / "table_spectrum.tex").write_text(table_tex + "\n")
    print(f"Saved {COMP_DIR / 'table_spectrum.tex'}")

    table_readable = render_readable_table(
        header_cells=["Wave family", "f_centroid (Hz)"], rows=rows_readable,
    )
    notes = "\n".join(summary_lines)
    (COMP_DIR / "table_spectrum_readable.txt").write_text(table_readable + "\n\n" + notes + "\n")
    print(table_readable)
    print(notes)
    print(f"Saved {COMP_DIR / 'table_spectrum_readable.txt'}")


if __name__ == "__main__":
    main()
