#!/usr/bin/env python3
# One-off schematic diagrams needed by the thesis but never drawn anywhere
# (no .pptx/.svg/.drawio/matplotlib source existed for any of them -- see
# the thesis-completion plan). Pure illustrative box/arrow schematics, not
# data analysis -- each function is self-contained and writes to an
# explicit absolute path rather than one shared output folder, since these
# diagrams are distributed across the phase folders they each belong to
# (situation/dataset -> dataset_examples/, MLP -> p0_baseline/, stencil ->
# p2_analysis/, CNN -> p14_MLP_vs_CNN/, teacher-forcing -> p0_baseline/,
# pushforward -> p1_pushforward/, TBPTT -> p1_bptt/), matching where every
# other figure for that phase already lives.
#
# Numeric labels are pulled from real config values (p0_baseline/p2_reference/
# p14_cnn's actual HIDDEN_SIZES/CNN_CHANNELS/SS/M_BACK/N_FWD/ndt/CFL), not
# invented -- these are schematics of the REAL configuration, not generic
# placeholders.
#
# Usage: python analysis/thesis_diagrams.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

BOX_KW = dict(boxstyle="round,pad=0.3", facecolor="#eaf1fb", edgecolor="#2a78d6", linewidth=1.5)
GT_BOX_KW = dict(boxstyle="round,pad=0.3", facecolor="#fdeee7", edgecolor="#eb6834", linewidth=1.5)
LOSS_BOX_KW = dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#333333", linewidth=1.5)


def _box(ax, xy, text, kw=BOX_KW, fontsize=10):
    ax.annotate(text, xy, ha="center", va="center", fontsize=fontsize, bbox=kw, zorder=3)


def _arrow(ax, xy_from, xy_to, style="-|>", color="#333333", ls="-", lw=1.6, connectionstyle="arc3,rad=0"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                                 connectionstyle=connectionstyle), zorder=2)


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
def plot_situation_diagram(out_path: Path):
    Nx = 100
    fig, ax = plt.subplots(figsize=(10, 3.0))
    x = np.linspace(0, 1, Nx)
    ax.plot(x, np.zeros_like(x), "-", color="#999999", lw=1, zorder=1)
    ax.scatter(x[1:-1], np.zeros(Nx - 2), s=18, color="#2a78d6", zorder=2, label="interior nodes")
    ax.scatter(x[[0, -1]], [0, 0], s=60, color="#eb6834", marker="s", zorder=3, label="boundary nodes")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, 0.8)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["x = 0", "x = L"])
    ax.set_title(f"Beam discretisation -- $N_x$ = {Nx} nodes", fontsize=12, pad=45)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.32))
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
def plot_mlp_architecture(out_path: Path, ss=11, m_back=2, n_fwd=3, hidden=(512, 256, 64)):
    input_dim = m_back * (2 * ss + 1)
    layer_sizes = [input_dim, *hidden, n_fwd]
    layer_labels = [f"Input\n{m_back}×(2×{ss}+1)={input_dim}"] + \
                   [f"Hidden\n{h}" for h in hidden] + [f"Output\n$N_{{\\mathrm{{FWD}}}}$={n_fwd}"]
    n_layers = len(layer_sizes)
    max_shown = 8
    fig, ax = plt.subplots(figsize=(2.2 * n_layers, 5))
    xs = np.arange(n_layers) * 2.0
    node_ys = {}
    for i, (size, label) in enumerate(zip(layer_sizes, layer_labels)):
        shown = min(size, max_shown)
        ys = np.linspace(-shown / 2, shown / 2, shown) * 0.5
        node_ys[i] = ys
        for y in ys:
            ax.add_patch(mpatches.Circle((xs[i], y), 0.12, facecolor="#2a78d6", edgecolor="#14345c", zorder=3))
        if size > max_shown:
            ax.annotate("⋮", (xs[i], ys[-1] + 0.4), ha="center", fontsize=14, zorder=3)
        ax.annotate(label, (xs[i], -shown * 0.5 * 0.5 - 0.9), ha="center", fontsize=9)
    for i in range(n_layers - 1):
        for y0 in node_ys[i]:
            for y1 in node_ys[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y0, y1], color="#c8c8c8", lw=0.5, zorder=1)
    ax.set_xlim(xs[0] - 1, xs[-1] + 1)
    ax.set_ylim(-4, 3)
    ax.axis("off")
    ax.set_title("MLP architecture (fully connected, ReLU hidden layers)", fontsize=12)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
def plot_cnn_architecture(out_path: Path, ss=11, m_back=2, n_fwd=3, channels=(16, 32), kernel=5):
    length = m_back * (2 * ss + 1)
    stages = [(1, length, "Input"), *[(c, length, f"Conv1d k={kernel}") for c in channels],
              (1, n_fwd, "Output (flatten+linear)")]
    fig, ax = plt.subplots(figsize=(2.4 * len(stages), 4))
    x = 0.0
    prev_center = None
    for i, (c, length_i, label) in enumerate(stages):
        w, h = 1.0, min(2.5, 0.3 + 0.15 * c)
        rect = mpatches.FancyBboxPatch((x, -h / 2), w, h, boxstyle="round,pad=0.05",
                                        facecolor="#eaf1fb", edgecolor="#2a78d6", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.annotate(f"{label}\n{c} ch × {length_i}", (x + w / 2, 0), ha="center", va="center", fontsize=9, zorder=4)
        center = (x + w / 2, 0)
        if prev_center is not None:
            _arrow(ax, (prev_center[0] + 0.5, 0), (x, 0))
        prev_center = center
        x += 2.0
    ax.set_xlim(-0.5, x)
    ax.set_ylim(-2, 2)
    ax.axis("off")
    ax.set_title("CNN (Conv1d) architecture -- shared kernel slides over the spatial stencil", fontsize=12)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
def plot_teacher_forcing(out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    _box(ax, (0.5, 1.5), "Reference solution\n$u^{n-1}, u^n$ (ground truth)", GT_BOX_KW)
    _box(ax, (3.0, 1.5), "Network $f_\\theta$")
    _box(ax, (5.5, 1.5), "Prediction\n$\\hat{u}^{n+1}$")
    _box(ax, (5.5, 0.0), "Reference $u^{n+1}$\n(ground truth)", GT_BOX_KW)
    _box(ax, (8.0, 0.75), "Loss $\\mathcal{L}_1$", LOSS_BOX_KW)
    _arrow(ax, (1.5, 1.5), (2.2, 1.5))
    _arrow(ax, (3.8, 1.5), (4.6, 1.5))
    _arrow(ax, (6.3, 1.5), (7.3, 0.95))
    _arrow(ax, (6.3, 0.0), (7.3, 0.55))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1.0, 2.5)
    ax.axis("off")
    ax.set_title("Teacher forcing -- every input drawn from the reference solution", fontsize=12)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
def plot_pushforward(out_path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    _box(ax, (0.3, 1.5), "$u^n$ (ground truth)", GT_BOX_KW)
    _box(ax, (2.6, 1.5), "Network $f_\\theta$\n(no_grad)")
    _box(ax, (5.0, 1.5), "$\\hat{u}^{n+1}$\n(detached, corrupted)")
    _box(ax, (7.4, 1.5), "Network $f_\\theta$\n(grad enabled)")
    _box(ax, (9.8, 1.5), "$\\hat{u}^{n+2}$")
    _box(ax, (9.8, 0.0), "$u^{n+2}$\n(ground truth)", GT_BOX_KW)
    _box(ax, (11.8, 0.75), "Loss $\\mathcal{L}_{pf}$", LOSS_BOX_KW)
    _arrow(ax, (1.2, 1.5), (1.7, 1.5))
    _arrow(ax, (3.6, 1.5), (4.1, 1.5), ls="--", color="#888888")
    _arrow(ax, (6.1, 1.5), (6.5, 1.5))
    _arrow(ax, (8.5, 1.5), (8.9, 1.5))
    _arrow(ax, (10.7, 1.5), (11.2, 0.95))
    _arrow(ax, (10.7, 0.0), (11.2, 0.55))
    ax.annotate("gradient flows through this application only", (7.4, 2.3), ha="center", fontsize=8.5,
                color="#555555", style="italic")
    ax.annotate("stop-gradient", (5.0, 2.15), ha="center", fontsize=8.5, color="#888888", style="italic")
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(-1.0, 2.7)
    ax.axis("off")
    ax.set_title("Pushforward trick -- first hop detached, second hop carries the gradient", fontsize=12)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
def plot_tbptt(out_path: Path, k=4):
    fig, ax = plt.subplots(figsize=(2.6 * k, 3.6))
    xs = [i * 2.6 for i in range(k)]
    _box(ax, (xs[0] - 1.3, 1.5), "$u^n$\n(ground truth)", GT_BOX_KW, fontsize=9)
    prev = xs[0] - 1.3
    for i, x in enumerate(xs):
        _box(ax, (x, 1.5), f"$f_\\theta$", fontsize=10)
        _arrow(ax, (prev + 0.6, 1.5), (x - 0.35, 1.5))
        pred_x = x + 1.0
        _box(ax, (pred_x, 1.5), f"$\\hat{{u}}^{{n+{i+1}}}$", fontsize=9)
        _arrow(ax, (x + 0.35, 1.5), (pred_x - 0.55, 1.5))
        _box(ax, (pred_x, 0.1), f"$u^{{n+{i+1}}}$\n(g.t.)", GT_BOX_KW, fontsize=8)
        prev = pred_x
    ax.annotate("", xy=(xs[0] - 0.3, 2.15), xytext=(xs[-1] + 0.3, 2.15),
                arrowprops=dict(arrowstyle="-|>", color="#b2182b", lw=1.8, connectionstyle="arc3,rad=-0.15"))
    ax.annotate("gradient backpropagated through all K applications", (sum(xs) / k, 2.55), ha="center",
                fontsize=9, color="#b2182b", style="italic")
    ax.set_xlim(xs[0] - 2.0, xs[-1] + 2.0)
    ax.set_ylim(-0.9, 3.0)
    ax.axis("off")
    ax.set_title(f"Truncated BPTT -- network applied K={k} times, loss accumulated over the window", fontsize=12)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
def plot_stencil_receptive_field(out_path: Path, ss=11, m_back=2, n_fwd=3, ndt=3, cfl=0.70):
    fig, ax = plt.subplots(figsize=(9, 6))
    # past input rows: t=0 (oldest of M_BACK) .. t=m_back-1, at x in [-ss, ss]
    xs = np.arange(-ss, ss + 1)
    for m in range(m_back):
        ax.scatter(xs, [m] * len(xs), s=22, color="#2a78d6", zorder=3)
    # future target rows: N_FWD hops, each ndt solver-steps apart, starting after m_back
    for f in range(1, n_fwd + 1):
        t = m_back - 1 + f * ndt
        ax.scatter([0], [t], s=60, color="#eb6834", marker="s", zorder=3)
    t_top = m_back - 1 + n_fwd * ndt
    # characteristic lines from the input window's edges (dashed lines, slope 1/CFL in this t-vs-x layout)
    t_line = np.linspace(m_back - 1, t_top, 50)
    ax.plot(-ss + (t_line - (m_back - 1)) * cfl, t_line, "--", color="#333333", lw=1.3)
    ax.plot(ss - (t_line - (m_back - 1)) * cfl, t_line, "--", color="#333333", lw=1.3)
    ax.axvspan(-ss, ss, ymin=0, ymax=(m_back - 1) / (t_top + 1), color="#2a78d6", alpha=0.05)
    ax.set_xlabel("spatial node offset from target (i)")
    ax.set_ylabel("time step")
    ax.set_yticks(range(0, t_top + 1))
    reach = n_fwd * ndt * cfl
    ax.set_title(
        f"Stencil receptive field: $M_{{\\mathrm{{BACK}}}}$={m_back}, $N_{{\\mathrm{{FWD}}}}$={n_fwd}, "
        f"$n_{{dt}}$={ndt}, SS={ss}, C≈{cfl:.2f}\n"
        f"characteristic reach = $N_{{\\mathrm{{FWD}}}} n_{{dt}} C$ = {reach:.1f} $\\leq$ SS = {ss}",
        fontsize=11)
    ax.grid(True, alpha=0.3)
    blue = mpatches.Patch(color="#2a78d6", label="input window (past states)")
    orange = mpatches.Patch(color="#eb6834", label="predicted increments (future)")
    ax.legend(handles=[blue, orange], loc="upper right", fontsize=9)
    _save(fig, out_path)


def plot_rod_assembly_structure(out_path: Path):
    # Real topology, not a schematic: reuses p7_structure_several_rods.py's
    # own build_lattice() so this figure is exactly the 94-rod lattice every
    # p7 assembly result is actually computed on, not an idealized drawing.
    import sys
    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    from p7_structure_several_rods import build_lattice
    from beamsurrogate.config import load_config

    cfg = load_config(REPO_ROOT / "runs/p7/p7_model_optimization_for_assembly/p7_dataset_medium_pinn_0.01/config.yaml")
    lattice = build_lattice(cfg)
    pos, edges, fixed, driven = lattice["pos"], lattice["edges"], lattice["fixed"], lattice["driven"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], "-", color="#333333", lw=1.2, zorder=1)
    for node_id, (x, y) in pos.items():
        if node_id in fixed:
            ax.scatter(x, y, s=40, color="#999999", zorder=3)
        elif node_id in driven:
            ax.scatter(x, y, s=55, color="#eb6834", zorder=3)
        else:
            ax.scatter(x, y, s=25, color="#2a78d6", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{len(edges)}-rod triangular lattice ({len(pos)} junction nodes) -- "
                 "each rod is its own beam, sharing state with neighbors at junctions", fontsize=12)
    fixed_p = mpatches.Patch(color="#999999", label="fixed (clamped) node")
    driven_p = mpatches.Patch(color="#eb6834", label="driven node")
    free_p = mpatches.Patch(color="#2a78d6", label="free junction node")
    ax.legend(handles=[fixed_p, driven_p, free_p], loc="upper center", ncol=3, frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, 1.02))
    _save(fig, out_path)


def main():
    plot_situation_diagram(REPO_ROOT / "runs/dataset_examples/figures/situation_diagram.png")
    plot_mlp_architecture(REPO_ROOT / "runs/p0/p0_baseline/figures/mlp_architecture.png")
    plot_cnn_architecture(REPO_ROOT / "runs/p14_MLP_vs_CNN/figures/CNN_architecture_explained.png")
    plot_teacher_forcing(REPO_ROOT / "runs/p0/p0_baseline/figures/teacher_forcing_training_strategy.png")
    plot_pushforward(REPO_ROOT / "runs/p1_pushforward/figures/training_diagram_pushforward.png")
    plot_tbptt(REPO_ROOT / "runs/p1_bptt/figures/training_diagram_tbptt.png")
    plot_stencil_receptive_field(REPO_ROOT / "runs/p2_analysis/figures/stencil_receptive_field.png")
    plot_rod_assembly_structure(REPO_ROOT / "runs/p7/p7_assembly_of_rods/figures/structure_assembling_of_rods.png")
    print("\nDone -- 8 diagrams generated.")


if __name__ == "__main__":
    main()
