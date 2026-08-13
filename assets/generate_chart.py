# -*- coding: utf-8 -*-
"""Generate assets/runtime-29x.png - measured LLM generation throughput on the
AGX Orin 64GB bench. 3D extruded bar style (matches speculative-by-task.png):
full green above the single-stream bandwidth ceiling, light green below, the
ceiling shown as a shaded zone with a marker under the axis.

All figures are this repo's own measurements (2026-08-05), proof directories
under benchmarks/.

Run from the repo root:
    uv run --with matplotlib python assets/generate_chart.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.patheffects import withStroke

OUT = os.path.dirname(os.path.abspath(__file__))

CEILING = 41.6  # 204.8 GB/s / 4.92 GB (Q4_K_M), single-stream

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

# 3D palettes (same as speculative-by-task.png)
ABOVE = {"front": "#76B900", "top": "#9BD434", "side": "#578A00"}
BELOW = {"front": "#AECF66", "top": "#CBE29E", "side": "#8FAF4F"}

DX3, DY3 = 1.0, 0.13   # isometric extrusion depth
H3 = 0.55              # bar height

rows = [
    ("30W power mode, stock clocks\nLlama 3.1 8B Q4_K_M", 8.1, "8.1 tok/s"),
    ("MAXN + locked clocks, JetPack 6.0\nLlama 3.1 8B Q4_K_M", 26.8, "26.8 tok/s"),
    ("Same, JetPack 6.2.1 (CUDA 12.6, rebuilt)\nLlama 3.1 8B Q4_K_M", 27.1, "27.1 tok/s"),
    ("Smaller quant of the same model\nLlama 3.1 8B IQ4_XS", 34.5, "34.5 tok/s"),
    ("Speculative decoding, 1B draft\nLlama 3.1 8B Q4_K_M + Llama 3.2 1B", 43.0, "43.0 tok/s"),
]
labels = [r[0] for r in rows]
values = [r[1] for r in rows]
vtexts = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, v in enumerate(values):
    pal = ABOVE if v > CEILING else BELOW
    y0, y1 = i - H3 / 2, i + H3 / 2
    ax.add_patch(Rectangle((0, y0), v, H3, facecolor=pal["front"],
                           edgecolor="none", zorder=3))
    ax.add_patch(Polygon([(0, y1), (DX3, y1 + DY3), (v + DX3, y1 + DY3),
                          (v, y1)], facecolor=pal["top"], edgecolor="none",
                         zorder=3))
    ax.add_patch(Polygon([(v, y0), (v + DX3, y0 + DY3), (v + DX3, y1 + DY3),
                          (v, y1)], facecolor=pal["side"], edgecolor="none",
                         zorder=3))

for i, (v, t) in enumerate(zip(values, vtexts)):
    ax.text(v + DX3 + 0.5, i, t, va="center", ha="left", fontsize=10.5,
            color=INK, fontweight="bold", zorder=6,
            path_effects=[withStroke(linewidth=3, foreground="white")])

ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels, fontsize=8.3, color=INK)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=0)

ax.set_xlim(0, 56)
ax.set_ylim(-0.55, 5.35)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left", "bottom"):
    ax.spines[spine].set_visible(False)

# Bandwidth ceiling: shaded zone in the plot, marker under the axis
# (no heavy vertical line - only speculative escapes the gray zone)
ax.axvspan(0, CEILING, color="#F1F3F5", zorder=0)
ax.axvline(CEILING, color="#CBD2D9", linewidth=1.0, zorder=1)

# mini-legend for the light/full semantics (empty bottom-right corner)
ax.add_patch(Rectangle((42.0, 0.50), 1.8, 0.24, facecolor=ABOVE["front"],
                       edgecolor="none", zorder=5))
ax.text(44.4, 0.62, "beats the ceiling", fontsize=7.8, color=INK,
        ha="left", va="center")
ax.add_patch(Rectangle((42.0, 0.14), 1.8, 0.24, facecolor=BELOW["front"],
                       edgecolor="none", zorder=5))
ax.text(44.4, 0.26, "below the ceiling", fontsize=7.8, color=MUTED,
        ha="left", va="center")

fig.suptitle("AGX Orin 64GB, measured: 8.1 to 43.0 tok/s on the same 8B model",
             fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left")
ax.set_title("Generation throughput (tokens/s), configuration in each label",
             fontsize=9, color=MUTED, loc="left", pad=12)

fig.text(0.02, 0.015,
         "All figures measured 2026-08-05 with this repo's scripts, full proofs in benchmarks/"
         "   |   Julien Becirovski",
         fontsize=7.5, color=MUTED)

fig.subplots_adjust(left=0.42, right=0.97, top=0.83, bottom=0.20)

# ceiling marker under the axis: triangle at the exact position + label
fx = fig.transFigure.inverted().transform(
    ax.transData.transform((CEILING, 0)))[0]
fig.text(fx, 0.115, "▲", fontsize=9, color=INK, ha="center", va="center")
fig.text(0.97, 0.075, "single-stream bandwidth ceiling (Q4_K_M): 41.6 tok/s",
         fontsize=8, color=MUTED, ha="right", va="center", fontweight="bold")

path = os.path.join(OUT, "runtime-29x.png")
fig.savefig(path, facecolor="white")
print("OK:", path)
