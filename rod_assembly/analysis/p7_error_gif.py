#!/usr/bin/env python3
# Top-down 2D animation of the |FD - NN| error field on the 94-rod lattice,
# the missing companion to the `<run_id>_fd_2d.gif` / `<run_id>_nn_2d.gif`
# pair that p7_structure_several_rods.py's save_gif() already writes.
#
# Why this exists rather than another flag on p7_structure_several_rods.py:
# that script has no "figures from cache" entry point -- calling it re-runs
# the whole FD + NN rollout. This reads the `<run_id>_frames.npz` cache it
# already left behind and recomputes no physics at all, same as
# p7_export_latex_report.py does.
#
# Note there is ALSO a `<run_id>_err_2d.gif` from that script: signed
# FD - NN on a diverging coolwarm map. Different figure, different purpose.
# That one answers "does the NN over- or under-predict here", but spends
# most of its frames at the neutral midpoint because the median |error| is
# ~0.2% of the shared colour range. This one drops the sign and plots
# |FD - NN| on the same sequential Reds/0..U_MAX scale as the fd/nn gifs,
# so the three can be read side by side as one set: the error frame is
# directly comparable to the displacement frames at the same instant.
#
# --own-scale rescales to the error's own maximum instead. Use it to see the
# error's spatial structure; the default shared scale is the one that keeps
# the three gifs comparable, and deliberately shows the error as faint
# because it IS small next to the signal.
#
# Usage: python analysis/p7_error_gif.py <run_id> [--format mp4|gif] [--own-scale]
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
import p7_structure_several_rods as m  # noqa: E402

MODELS_DIR = REPO_ROOT / "rod_assembly" / "runs" / "models"
PAD = 0.4  # same lattice framing as save_gif()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--own-scale", action="store_true",
                        help="scale the colourbar to the error's own max instead of the "
                             "U_MAX shared with the fd/nn gifs (breaks comparability)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4",
                        help="mp4 (default): h264, far smaller than the gif and seekable in "
                             "a player. gif: the palette-quantised loop, for anything that "
                             "cannot embed video.")
    parser.add_argument("--dpi", type=int, default=100,
                        help="100 -> 750x420, the same pixel size as the fd/nn gifs")
    args = parser.parse_args()

    run_dir = MODELS_DIR / args.run_id
    cache_path = run_dir / f"{args.run_id}_frames.npz"
    if not cache_path.exists():
        sys.exit(f"No cached frames at {cache_path} -- run "
                 f"p7_structure_several_rods.py {args.run_id} first.")

    cfg, _src_run_dir = m.load_source(args.run_id)
    lattice = m.build_lattice(cfg)
    edges, pos = lattice["edges"], lattice["pos"]

    cached = np.load(cache_path)
    fd_steps, fd_U = cached["fd_steps"], cached["fd_U"]
    nn_steps, nn_U = cached["nn_steps"], cached["nn_U"]

    # Same overlap-based FD/NN alignment as compute_metrics()/_make_figures():
    # the NN rollout starts M_BACK steps in, so the two step vectors only
    # partially overlap and must be intersected, not sliced.
    err_steps, fd_idx, nn_idx = np.intersect1d(fd_steps, nn_steps, assume_unique=True,
                                               return_indices=True)
    abs_err = np.abs(fd_U[fd_idx] - nn_U[nn_idx])

    # U_MAX is computed over all three fields exactly as _make_figures() does,
    # so this gif's colour scale is bit-for-bit the one the fd/nn gifs used.
    U_MAX = float(max(np.abs(fd_U).max(), np.abs(nn_U).max(), abs_err.max()))
    vmax = float(abs_err.max()) if args.own_scale else U_MAX

    XS = np.concatenate([np.linspace(pos[a][0], pos[b][0], cfg.Nx) for a, b in edges])
    YS = np.concatenate([np.linspace(pos[a][1], pos[b][1], cfg.Nx) for a, b in edges])

    fig = plt.figure(figsize=(7.5, 4.2))
    ax = fig.add_axes([0.02, 0.04, 0.82, 0.86])
    ax.set_xlim(XS.min() - PAD, XS.max() + PAD)
    ax.set_ylim(YS.min() - PAD, YS.max() + PAD)
    ax.set_aspect("equal")
    ax.set_axis_off()
    scat = ax.scatter(XS, YS, c=np.zeros_like(XS), cmap="Reds", vmin=0.0, vmax=vmax, s=7)
    cax = fig.add_axes([0.875, 0.30, 0.018, 0.42])
    cbar = fig.colorbar(scat, cax=cax)
    cbar.set_label("|FD - NN|", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    title = fig.suptitle("", fontsize=10, y=0.955)

    scale_note = "own scale" if args.own_scale else "shared FD/NN scale"
    label = (f"gaussian pulse, |FD - NN| error ({args.run_id}), {scale_note}")

    def update(i):
        scat.set_array(abs_err[i].ravel())
        title.set_text(f"{label}  --  t = {err_steps[i] * cfg.dt:.2f} s  (top view)")
        return [scat, title]

    stem = "_abs_err_2d_ownscale" if args.own_scale else "_abs_err_2d"
    out_path = run_dir / "figures" / f"{args.run_id}{stem}.{args.format}"
    anim = animation.FuncAnimation(fig, update, frames=len(err_steps), interval=40)
    if args.format == "mp4":
        # yuv420p because players that reject 4:4:4 are the common case
        # (QuickTime, PowerPoint, most browsers); the pad filter rounds odd
        # pixel dimensions up to even, which libx264 requires in that pixel
        # format and which any --dpi other than 100 will otherwise produce.
        writer = animation.FFMpegWriter(
            fps=args.fps, codec="libx264", bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    else:
        writer = animation.PillowWriter(fps=args.fps)
    anim.save(out_path, writer=writer, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  frames        : {len(err_steps)}  "
          f"(t = {err_steps[0] * cfg.dt:.2f}s .. {err_steps[-1] * cfg.dt:.2f}s)")
    print(f"  colour scale  : 0 .. {vmax:.6f}   ({scale_note}; U_MAX = {U_MAX:.6f})")
    print(f"  |err| max     : {abs_err.max():.6f}   median : {np.median(abs_err):.3e}")


if __name__ == "__main__":
    main()
