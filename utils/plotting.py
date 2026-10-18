"""Shared figure style and colour assignments.

Figures are drawn at their printed size (PRL/revtex: one column = 3.4 in, full
text width = 7.06 in), so the font sizes below are the sizes on the page. They
sit just under the 9 pt captions and 10 pt body text.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot is imported
import matplotlib.pyplot as plt

from . import config

COLUMN_WIDTH = 3.4   # in
TEXT_WIDTH = 7.06    # in

# Colours follow the entity throughout the paper, using the matplotlib cycle C0, C1, ...
COLORS = {
    "bs": "C0",                # boson-star signal
    "bh": "C3",                # binary-BH comparison waveform / model
    "data": "darkgrey",        # noisy data
    "insp40": "C0",            # t_< = 40 T_M
    "insp90": "C3",            # t_< = 90 T_M
    "full": "C0",              # full-signal analysis
    "tlt": "C3",               # t_< markers
    "tgt": "C2",               # t_> markers
    "mode1": "C0",             # 220 model
    "mode2": "C3",             # 220+221 model
}
INSPIRAL_SHORT = {40: r"$t_<$ = 40 $T_M$", 90: r"$t_<$ = 90 $T_M$"}
LABEL_SIZE = 8      # pt
TICK_SIZE = 7       # pt
LEGEND_SIZE = 6.5   # pt
GRID = dict(visible=True, alpha=0.3)


def use_style():
    """Reset matplotlib and apply the paper's font sizes, line widths and PDF settings."""
    plt.rcdefaults()
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": TICK_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "legend.handlelength": 1.8,
    })


def save(fig, name):
    """Save fig to figures/<name>, close it and return the path.

    Set the environment variable FIG_PREVIEW_DIR to a directory to also write a PNG preview there.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / name
    fig.savefig(path)
    if os.environ.get("FIG_PREVIEW_DIR"):  # optional PNG previews
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / path.with_suffix(".png").name, dpi=200)
    plt.close(fig)
    print(f"wrote {path.relative_to(config.REPO)}")
    return path
