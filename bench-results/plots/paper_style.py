"""Shared matplotlib style for every figure that goes into the paper.

The paper is typeset in Times (usenix-2020-09.sty loads mathptmx), so
figure text uses Times as well, with STIX for math, which is the
Times-compatible math font that ships with matplotlib. Import this
module once, right after matplotlib.pyplot, in each plot_*.py.
"""
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    # First match wins; the fallbacks cover Linux boxes without the
    # Microsoft core fonts.
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # Embed fonts as TrueType (Type 42) instead of Type 3, which some
    # camera-ready checkers reject.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Figures are drawn at the width they occupy in the paper, so that a
# font size set here is the size the reader sees. A USENIX column is
# 3.33 in wide; drawing a figure wider and letting LaTeX shrink it also
# shrinks its text (a 6.5 in figure with 7 pt labels prints at 3.6 pt).
COLUMN_WIDTH_IN = 3.33
LABEL_PT = 7      # axis labels and legends
TICK_PT = 6       # tick labels
