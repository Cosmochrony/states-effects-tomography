"""
Generates two figures, each placed inline near its relevant section rather than bundled in the appendix.

fig_cv1.pdf illustrates Theorem 2.2 (Section 2): the finite family of cylindrical conditionings on
Delta(S_A), m=3, reaches exactly 7 = 2^3-1 points (the 3 Dirac vertices, the 3 two-element-subset
conditionings on the edges, and the unconditioned prior in the interior) -- while the shaded triangle is
the full simplex, reachable only once the classical-randomisation postulate P1 (Axiom 3.1) is added.

fig_diagonal.pdf illustrates Theorems 7.1-7.2 (Section 7): on the joint outcome set S_A x S_B = {0,1}^2, a
rectangle (e.g. the column {0,1}x{0}) is reachable as a product of local sharp effects, while the diagonal
{(0,0),(1,1)} is not reachable by any combination of local products, however locally randomised (the
CHSH-type witness of Theorem 7.2).

Both figures are illustrative of already-proven finite/combinatorial facts (see the six code/*_tests.py
scripts for the exact verification); neither introduces a new numerical claim.

Deterministic: no randomness; SOURCE_DATE_EPOCH is honoured by the build script for reproducible PDF
metadata.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

OUT_DIR = "out"


def barycentric_to_cartesian(p0, p1, p2):
    # Vertices: e_0 -> (0,0), e_1 -> (1,0), e_2 -> (0.5, sqrt(3)/2)
    x = p1 * 1.0 + p2 * 0.5
    y = p2 * (math.sqrt(3) / 2)
    return x, y


def panel_a(ax):
    v0 = barycentric_to_cartesian(1, 0, 0)
    v1 = barycentric_to_cartesian(0, 1, 0)
    v2 = barycentric_to_cartesian(0, 0, 1)
    triangle = Polygon([v0, v1, v2], closed=True, facecolor="#cdd7f0", edgecolor="black",
                        alpha=0.55, zorder=1)
    ax.add_patch(triangle)

    alpha = (0.5, 1.0 / 3.0, 1.0 / 6.0)
    # 7 achieved points: 3 Diracs, 3 two-subset conditionings (edges), 1 full/prior (interior).
    points = {
        r"$\delta_0$": (1, 0, 0),
        r"$\delta_1$": (0, 1, 0),
        r"$\delta_2$": (0, 0, 1),
    }
    s01 = alpha[0] + alpha[1]
    s02 = alpha[0] + alpha[2]
    s12 = alpha[1] + alpha[2]
    points[r"$p_{\{0,1\}}$"] = (alpha[0] / s01, alpha[1] / s01, 0)
    points[r"$p_{\{0,2\}}$"] = (alpha[0] / s02, 0, alpha[2] / s02)
    points[r"$p_{\{1,2\}}$"] = (0, alpha[1] / s12, alpha[2] / s12)
    points[r"$\alpha$ (prior)"] = alpha

    for label, (p0, p1, p2) in points.items():
        x, y = barycentric_to_cartesian(p0, p1, p2)
        ax.plot(x, y, "o", color="#1a2b6d", markersize=7, zorder=3)
        offset = (0.03, 0.03) if label != r"$\alpha$ (prior)" else (0.05, -0.02)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(offset[0] * 100, offset[1] * 100),
                    fontsize=9, zorder=4)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"$\Delta(S_A)$, $m=3$: 7 achieved points vs. the shaded full simplex",
                 fontsize=10)


def panel_b(ax):
    # Two 2x2 grids side by side: left = rectangle (row), right = diagonal.
    def draw_grid(origin_x, highlighted_cells, title):
        for i in range(2):
            for j in range(2):
                x = origin_x + j * 1.0
                y = i * 1.0
                cell = (1 - i, j)  # row 0 at top visually
                filled = cell in highlighted_cells
                rect = Rectangle((x, y), 1, 1, facecolor="#f0a35a" if filled else "#e8e8e8",
                                  edgecolor="black", zorder=1)
                ax.add_patch(rect)
                ax.annotate(f"({cell[0]},{cell[1]})", (x + 0.5, y + 0.5), ha="center", va="center",
                            fontsize=8, zorder=2)
        ax.annotate(title, (origin_x + 1.0, -0.35), ha="center", fontsize=9)

    draw_grid(0.0, {(0, 0), (1, 0)}, "rectangle $\\{0,1\\}\\times\\{0\\}$\n(local product, reachable)")
    draw_grid(3.0, {(0, 0), (1, 1)}, "diagonal $\\{(0,0),(1,1)\\}$\n(not a product, unreachable)")

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.9, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rectangles vs. the diagonal on $S_A \\times S_B = \\{0,1\\}^2$", fontsize=10)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig_a, ax_a = plt.subplots(figsize=(5.2, 4.6))
    panel_a(ax_a)
    fig_a.tight_layout()
    path_a = os.path.join(OUT_DIR, "fig_cv1.pdf")
    fig_a.savefig(path_a, metadata={"CreationDate": None})
    print(f"Wrote {path_a}")

    fig_b, ax_b = plt.subplots(figsize=(6.4, 3.2))
    panel_b(ax_b)
    fig_b.tight_layout()
    path_b = os.path.join(OUT_DIR, "fig_diagonal.pdf")
    fig_b.savefig(path_b, metadata={"CreationDate": None})
    print(f"Wrote {path_b}")


if __name__ == "__main__":
    main()
