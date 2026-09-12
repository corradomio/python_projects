from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple
import random


# --------------------------------------------------------------------------- #
#  Data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
    w: float
    h: float
    id: int = -1

    @property
    def area(self) -> float:
        return self.w * self.h

    def rotated(self) -> "Rect":
        return Rect(self.h, self.w, self.id)


@dataclass
class Placement:
    id: int          # id of the small rectangle
    x: float         # left edge (relative to the big rectangle's bottom-left = (0,0))
    y: float         # bottom edge
    w: float         # width  as placed
    h: float         # height as placed
    rotated: bool    # True if the piece was rotated by 90°

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def __str__(self) -> str:
        r = " (rotated)" if self.rotated else ""
        return f"rect {self.id}: {self.w:g}x{self.h:g} at ({self.x:.3f}, {self.y:.3f}){r}"


# --------------------------------------------------------------------------- #
#  Solver
# --------------------------------------------------------------------------- #
class RectangleCoverSolver:
    """
    Cover a big W x H rectangle with small rectangles.
      - rotation by 90° allowed
      - overlap allowed (surplus is distributed uniformly)
      - cutting NOT allowed
      - not every piece has to be used
    """

    ORIENTATIONS = ("landscape", "portrait", "as_is")
    SORT_KEYS = {
        "height": lambda r: (-r.h, -r.w),
        "width": lambda r: (-r.w, -r.h),
        "area": lambda r: (-r.area, -r.h),
    }

    def __init__(self, W: float, H: float, rects: Sequence[Tuple[float, float]],
                 eps: float = 1e-9):
        if W <= 0 or H <= 0:
            raise ValueError("big rectangle must have positive size")
        self.W, self.H, self.eps = float(W), float(H), eps
        self.rects = [Rect(float(w), float(h), i) for i, (w, h) in enumerate(rects)]
        if sum(r.area for r in self.rects) < W * H - eps:
            raise ValueError("total area of small rectangles is smaller than the big one")

    # ------------------------------------------------------------------ public
    def solve(self, random_restarts: int = 50, seed: int = 0) -> Optional[List[Placement]]:
        """
        Try all deterministic strategies (+ optional random restarts) and return
        the covering with the least total placed area (i.e. least overlap).
        Returns None if no strategy succeeds.
        """
        rng = random.Random(seed)
        best: Optional[List[Placement]] = None
        best_cost = float("inf")

        candidates = []
        for transposed, orient, key in product((False, True), self.ORIENTATIONS,
                                               self.SORT_KEYS):
            candidates.append((transposed, orient, key, None))
        for k in range(random_restarts):
            candidates.append((rng.random() < 0.5, rng.choice(self.ORIENTATIONS),
                               None, rng.random()))

        for transposed, orient, key, shuffle in candidates:
            W, H = (self.H, self.W) if transposed else (self.W, self.H)
            pieces = self._orient(self.rects, orient)
            if key is not None:
                pieces.sort(key=self.SORT_KEYS[key])
            else:
                rng.shuffle(pieces)
                pieces.sort(key=lambda r: -r.h)  # keep "tall first", random tie-break

            rows = self._build_rows(W, H, pieces)
            if rows is None:
                continue
            placements = self._layout(W, H, rows)
            if transposed:
                placements = [Placement(p.id, p.y, p.x, p.h, p.w, not p.rotated)
                              for p in placements]
            if not self.verify(placements):
                continue
            cost = sum(p.w * p.h for p in placements)
            if cost < best_cost:
                best, best_cost = placements, cost
        return best

    def verify(self, placements: List[Placement]) -> bool:
        """Exact check: every elementary cell of the big rectangle is covered."""
        e = self.eps
        xs = sorted({0.0, self.W} | {min(max(v, 0.0), self.W)
                                    for p in placements for v in (p.x, p.x2)})
        ys = sorted({0.0, self.H} | {min(max(v, 0.0), self.H)
                                    for p in placements for v in (p.y, p.y2)})
        for i in range(len(xs) - 1):
            cx = 0.5 * (xs[i] + xs[i + 1])
            if xs[i + 1] - xs[i] < e:
                continue
            for j in range(len(ys) - 1):
                if ys[j + 1] - ys[j] < e:
                    continue
                cy = 0.5 * (ys[j] + ys[j + 1])
                if not any(p.x - e <= cx <= p.x2 + e and p.y - e <= cy <= p.y2 + e
                           for p in placements):
                    return False
        return True

    # ---------------------------------------------------------------- helpers
    def _orient(self, rects: Sequence[Rect], policy: str) -> List[Rect]:
        out = []
        for r in rects:
            if policy == "landscape" and r.h > r.w:
                r = r.rotated()
            elif policy == "portrait" and r.w > r.h:
                r = r.rotated()
            out.append(r)
        return out

    def _build_rows(self, W: float, H: float,
                    pieces: List[Rect]) -> Optional[List[Tuple[List[Rect], float]]]:
        """
        Greedy shelf construction. Each row is a list of pieces whose widths sum
        to >= W; its 'band' (guaranteed covered height) is the min piece height.
        Returns list of (row_pieces, band) or None on failure.
        """
        e = self.eps
        remaining = list(pieces)
        rows, covered = [], 0.0

        while covered < H - e:
            if not remaining:
                return None
            row, width = [], 0.0
            while width < W - e and remaining:
                p = remaining.pop(0)
                row.append(p)
                width += p.w
            if width < W - e:
                return None

            # local improvement: rotate the shortest piece if that raises the band
            while True:
                band = min(p.h for p in row)
                i = min(range(len(row)), key=lambda k: row[k].h)
                p = row[i]
                rot = p.rotated()
                new_width = width - p.w + rot.w
                if rot.h > band + e and new_width >= W - e:
                    row[i], width = rot, new_width
                else:
                    break
            band = min(p.h for p in row)
            # drop pieces that are not needed to reach W (put back for later rows)
            row, width = self._trim_row(W, row, width, remaining)
            rows.append((row, band))
            covered += band
        return rows

    def _trim_row(self, W: float, row: List[Rect], width: float,
                  remaining: List[Rect]) -> Tuple[List[Rect], float]:
        """Remove pieces (smallest first) that are unnecessary to span W."""
        e = self.eps
        changed = True
        while changed and len(row) > 1:
            changed = False
            for p in sorted(row, key=lambda r: r.w):
                if width - p.w >= W - e:
                    row.remove(p)
                    width -= p.w
                    remaining.insert(0, p)
                    changed = True
                    break
        return row, width

    def _layout(self, W: float, H: float,
                rows: List[Tuple[List[Rect], float]]) -> List[Placement]:
        """Place rows with uniformly distributed overlaps (horizontal & vertical)."""
        placements = []
        bands = [band for _, band in rows]
        surplus_v = sum(bands) - H
        ov_v = surplus_v / (len(rows) - 1) if len(rows) > 1 else 0.0
        # a single band taller than H: centre it (equal overhang top and bottom)
        y = -surplus_v / 2 if len(rows) == 1 else 0.0

        for row, band in rows:
            width = sum(p.w for p in row)
            surplus_h = width - W
            ov_h = surplus_h / (len(row) - 1) if len(row) > 1 else 0.0
            x = -surplus_h / 2 if len(row) == 1 else 0.0
            for p in row:
                orig = self.rects[p.id]
                rotated = not (abs(orig.w - p.w) < self.eps and abs(orig.h - p.h) < self.eps)
                placements.append(Placement(p.id, x, y, p.w, p.h, rotated))
                x += p.w - ov_h
            y += band - ov_v
        return placements

    # ---------------------------------------------------------------- output
    def report(self, placements: Optional[List[Placement]]) -> str:
        if placements is None:
            return "No covering found."
        used = sum(p.w * p.h for p in placements)
        lines = [f"Big rectangle {self.W:g} x {self.H:g}  (area {self.W*self.H:g})",
                 f"Pieces used: {len(placements)} / {len(self.rects)}, "
                 f"placed area {used:g}, overlap+overhang {used - self.W*self.H:g}"]
        lines += [f"  {p}" for p in placements]
        return "\n".join(lines)

    # def plot(self, placements: List[Placement]) -> None:
    #     """Optional visualisation (requires matplotlib)."""
    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Rectangle
    #     fig, ax = plt.subplots()
    #     ax.add_patch(Rectangle((0, 0), self.W, self.H, fill=False, lw=3, ec="black"))
    #     for p in placements:
    #         ax.add_patch(Rectangle((p.x, p.y), p.w, p.h, alpha=0.35, ec="k"))
    #         ax.text(p.x + p.w / 2, p.y + p.h / 2, str(p.id), ha="center", va="center")
    #     margin = 0.1 * max(self.W, self.H)
    #     ax.set_xlim(-margin, self.W + margin)
    #     ax.set_ylim(-margin, self.H + margin)
    #     ax.set_aspect("equal")
    #     plt.show()
    def plot(self, placements: Optional[List[Placement]],
             show_overlap: bool = True,
             show_unused: bool = True,
             title: Optional[str] = None,
             savefile: Optional[str] = None,
             show: bool = True):
        """
        Visualise a covering with matplotlib.

        show_overlap : shade elementary cells covered by >= 2 pieces
                       (darker = more layers)
        show_unused  : draw the unused small rectangles in a side panel
        savefile     : if given, save the figure to this path (png, pdf, svg...)
        show         : call plt.show()
        Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Patch
        from matplotlib import colormaps

        if placements is None:
            raise ValueError("nothing to plot: placements is None")

        used_ids = {p.id for p in placements}
        unused = [r for r in self.rects if r.id not in used_ids] if show_unused else []

        if unused:
            fig, (ax, ax_un) = plt.subplots(
                1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3, 1]})
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax_un = None

        cmap = colormaps.get_cmap("tab20")
        n_colors = 20

        # ---- big rectangle ------------------------------------------------
        ax.add_patch(Rectangle((0, 0), self.W, self.H,
                               fill=False, lw=3, ec="black", zorder=5))

        # ---- placed pieces -----------------------------------------------
        for p in placements:
            color = cmap(p.id % n_colors)
            ax.add_patch(Rectangle((p.x, p.y), p.w, p.h,
                                   facecolor=color, alpha=0.45,
                                   edgecolor="black", lw=1, zorder=2))
            label = f"{p.id}" + ("↻" if p.rotated else "")
            ax.text(p.x + p.w / 2, p.y + p.h / 2, label,
                    ha="center", va="center", fontsize=9, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.7))

        # ---- overlap shading ---------------------------------------------
        if show_overlap:
            e = self.eps
            xs = sorted({0.0, self.W} | {min(max(v, 0.0), self.W)
                                        for p in placements for v in (p.x, p.x2)})
            ys = sorted({0.0, self.H} | {min(max(v, 0.0), self.H)
                                        for p in placements for v in (p.y, p.y2)})
            max_layers = 1
            cells = []
            for i in range(len(xs) - 1):
                if xs[i + 1] - xs[i] < e:
                    continue
                cx = 0.5 * (xs[i] + xs[i + 1])
                for j in range(len(ys) - 1):
                    if ys[j + 1] - ys[j] < e:
                        continue
                    cy = 0.5 * (ys[j] + ys[j + 1])
                    k = sum(1 for p in placements
                            if p.x - e <= cx <= p.x2 + e and p.y - e <= cy <= p.y2 + e)
                    if k >= 2:
                        cells.append((xs[i], ys[j], xs[i + 1] - xs[i],
                                      ys[j + 1] - ys[j], k))
                        max_layers = max(max_layers, k)
            for x, y, w, h, k in cells:
                alpha = 0.25 + 0.5 * (k - 2) / max(1, max_layers - 2)
                ax.add_patch(Rectangle((x, y), w, h, facecolor="red",
                                       edgecolor="none", alpha=min(alpha, 0.85),
                                       hatch="//", zorder=4))

        # ---- axes cosmetics ----------------------------------------------
        margin = 0.08 * max(self.W, self.H)
        xmin = min([0.0] + [p.x for p in placements]) - margin
        xmax = max([self.W] + [p.x2 for p in placements]) + margin
        ymin = min([0.0] + [p.y for p in placements]) - margin
        ymax = max([self.H] + [p.y2 for p in placements]) + margin
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.4)

        placed_area = sum(p.w * p.h for p in placements)
        if title is None:
            title = (f"Cover {self.W:g}×{self.H:g}  —  {len(placements)} pieces, "
                     f"placed area {placed_area:g} "
                     f"(+{placed_area - self.W * self.H:g} overlap/overhang)")
        ax.set_title(title, fontsize=10)

        handles = [Patch(fill=False, ec="black", lw=3, label="big rectangle"),
                   Patch(fc="grey", alpha=0.45, ec="black", label="placed piece (↻ = rotated)")]
        if show_overlap:
            handles.append(Patch(fc="red", alpha=0.4, hatch="//", label="overlap (≥2 layers)"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.05),
                  ncol=3, fontsize=8, frameon=False)

        # ---- unused pieces panel -----------------------------------------
        if ax_un is not None:
            y = 0.0
            gap = 0.05 * max(self.W, self.H)
            maxw = 0.0
            for r in unused:
                ax_un.add_patch(Rectangle((0, y), r.w, r.h,
                                          facecolor=cmap(r.id % n_colors),
                                          alpha=0.45, edgecolor="black"))
                ax_un.text(r.w / 2, y + r.h / 2, str(r.id),
                           ha="center", va="center", fontsize=9)
                y += r.h + gap
                maxw = max(maxw, r.w)
            ax_un.set_xlim(-gap, maxw + gap)
            ax_un.set_ylim(-gap, y)
            ax_un.set_aspect("equal")
            ax_un.set_title(f"unused pieces ({len(unused)})", fontsize=10)
            ax_un.axis("off")

        fig.tight_layout()
        if savefile:
            fig.savefig(savefile, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig


# --------------------------------------------------------------------------- #
#  Example
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    big_W, big_H = 10, 8
    small = [(6, 3), (5, 3), (4, 3), (7, 2), (4, 2), (3, 5), (2, 6), (5, 2)]
    small_2 = []

    for i in range(4):
        for x, y in small:
            rx = random.uniform(0.1, 0.9)
            ry = random.uniform(0.1, 0.9)
            small_2.append((x*rx, y*ry))
    small = small_2

    solver = RectangleCoverSolver(big_W, big_H, small)
    result = solver.solve(random_restarts=200)
    print(solver.report(result))
    solver.plot(result)   # uncomment if matplotlib is available