"""
rect_cover.py
=============

Cover a big rectangle with a collection of smaller rectangles.

The pieces may overlap each other, but the whole big rectangle must end up
covered, and (by default) nothing is allowed to stick out of it.

Main entry point
----------------

    solver = RectangleCover(width=100, height=60, rectangles=[(10, 8), (12, 5), ...])
    result = solver.solve()
    for p in result.placements:
        print(p.index, p.cx, p.cy, p.angle)      # centre + orientation (0 or 90)

Algorithm
---------
A *shelf* (horizontal band) construction.  The big rectangle is swept top to
bottom.  At each step the solver chooses a band height `t` and a subset of the
still-unused pieces such that every chosen piece is at least `t` tall (in the
orientation it is given) and their widths sum to at least the full width `W`.
Laid side by side and flushed to the top of the band, those pieces cover the
strip `[y, y + t]` completely.  The band height is then advanced and the
process repeats until the bottom edge is reached.

Among all candidate band heights the solver picks the one maximising
`height_gained / area_consumed`, i.e. the greedy that wastes the least material
per unit of progress.  The whole construction is run twice - once normally and
once on the transposed problem (vertical bands) - and the cheaper of the two
results is returned.

This is a heuristic: the exact problem (minimum number of rectangles covering a
rectangle) is NP-hard.  Failure of `solve()` therefore means "this solver could
not find a cover", not "no cover exists" - except when one of the necessary
conditions checked by `feasibility_report()` is violated, in which case no
cover exists at all.

Verification
------------
`CoverResult.verify()` re-checks the returned placements from scratch, by
coordinate compression, so the answer is never trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Placement",
    "CoverResult",
    "CoverError",
    "RectangleCover",
]

EPS = 1e-9


class CoverError(RuntimeError):
    """Raised when no covering could be produced."""


# --------------------------------------------------------------------------- #
#  Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Placement:
    """One small rectangle, placed.

    Attributes
    ----------
    index : int
        Position of the piece in the input list.
    cx, cy : float
        Coordinates of the **centre** of the piece.  The origin (0, 0) is the
        top-left corner of the big rectangle, x grows right, y grows down.
    angle : int
        Orientation in degrees: 0 = as given, 90 = rotated a quarter turn.
    width, height : float
        Footprint *after* the rotation has been applied (so `width` is always
        the horizontal extent).
    """

    index: int
    cx: float
    cy: float
    angle: int
    width: float
    height: float

    # -- convenience ------------------------------------------------------- #
    @property
    def rotated(self) -> bool:
        return self.angle == 90

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max)."""
        hw, hh = self.width / 2.0, self.height / 2.0
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Placement(index={self.index}, centre=({self.cx:g}, {self.cy:g}), "
                f"angle={self.angle}, size={self.width:g}x{self.height:g})")


@dataclass
class CoverResult:
    """Outcome of a successful :meth:`RectangleCover.solve` call."""

    width: float
    height: float
    placements: List[Placement] = field(default_factory=list)
    unused: List[int] = field(default_factory=list)

    # -- reporting --------------------------------------------------------- #
    @property
    def used_area(self) -> float:
        return sum(p.width * p.height for p in self.placements)

    @property
    def overlap_ratio(self) -> float:
        """Material spent divided by the area actually needed (1.0 = perfect)."""
        target = self.width * self.height
        return self.used_area / target if target else float("inf")

    def centers(self) -> List[Tuple[float, float, int]]:
        """The bare answer: [(cx, cy, angle), ...] in input order."""
        return [(p.cx, p.cy, p.angle) for p in sorted(self.placements, key=lambda q: q.index)]

    # -- validation -------------------------------------------------------- #
    def verify(self, inside_only: bool = True) -> bool:
        """Independently re-check that the big rectangle is fully covered.

        Uses coordinate compression: the union of the placed rectangles is
        covered iff every elementary cell of the grid induced by all the
        rectangle edges is covered, which is tested at the cell centre.
        """
        W, H = self.width, self.height
        boxes = [p.bounds for p in self.placements]

        if inside_only:
            for (x0, y0, x1, y1), p in zip(boxes, self.placements):
                if x0 < -EPS or y0 < -EPS or x1 > W + EPS or y1 > H + EPS:
                    return False

        xs = sorted({0.0, W} | {min(max(v, 0.0), W) for b in boxes for v in (b[0], b[2])})
        ys = sorted({0.0, H} | {min(max(v, 0.0), H) for b in boxes for v in (b[1], b[3])})

        for i in range(len(xs) - 1):
            if xs[i + 1] - xs[i] < EPS:
                continue
            mx = 0.5 * (xs[i] + xs[i + 1])
            for j in range(len(ys) - 1):
                if ys[j + 1] - ys[j] < EPS:
                    continue
                my = 0.5 * (ys[j] + ys[j + 1])
                if not any(x0 - EPS <= mx <= x1 + EPS and y0 - EPS <= my <= y1 + EPS
                           for (x0, y0, x1, y1) in boxes):
                    return False
        return True

    # -- drawing ----------------------------------------------------------- #
    def to_svg(self, scale: float = 1.0, margin: float = 20.0) -> str:
        """A stand-alone SVG string, handy for eyeballing the layout."""
        W, H = self.width * scale, self.height * scale
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{W + 2 * margin:g}" height="{H + 2 * margin:g}" '
            f'viewBox="{-margin:g} {-margin:g} {W + 2 * margin:g} {H + 2 * margin:g}">',
            f'<rect x="0" y="0" width="{W:g}" height="{H:g}" fill="#f4f4f2"/>',
        ]
        palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
                   "#b279a2", "#ff9da6", "#9d755d", "#eeca3b", "#79706e"]
        for k, p in enumerate(self.placements):
            x0, y0, x1, y1 = p.bounds
            colour = palette[k % len(palette)]
            parts.append(
                f'<rect x="{x0 * scale:g}" y="{y0 * scale:g}" '
                f'width="{(x1 - x0) * scale:g}" height="{(y1 - y0) * scale:g}" '
                f'fill="{colour}" fill-opacity="0.45" stroke="{colour}" stroke-width="1"/>'
            )
            parts.append(
                f'<circle cx="{p.cx * scale:g}" cy="{p.cy * scale:g}" r="1.8" fill="#111"/>'
            )
        parts.append(
            f'<rect x="0" y="0" width="{W:g}" height="{H:g}" '
            f'fill="none" stroke="#111" stroke-width="2"/>'
        )
        parts.append("</svg>")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
#  Solver
# --------------------------------------------------------------------------- #

@dataclass
class _Piece:
    index: int
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


class RectangleCover:
    """Cover a `width` x `height` rectangle with the given small rectangles.

    Parameters
    ----------
    width, height : float
        Dimensions of the big rectangle to be covered.
    rectangles : sequence of (w, h)
        The available pieces.  Not all of them need to be used.
    allow_rotation : bool, default True
        Whether a piece may be turned by 90 degrees.
    """

    def __init__(self,
                 width: float,
                 height: float,
                 rectangles: Iterable[Sequence[float]],
                 allow_rotation: bool = True,
                 allow_overhang: bool = False) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("the big rectangle must have positive dimensions")

        self.width = float(width)
        self.height = float(height)
        self.allow_rotation = bool(allow_rotation)
        self.allow_overhang = bool(allow_overhang)

        self.pieces: List[_Piece] = []
        for i, wh in enumerate(rectangles):
            w, h = float(wh[0]), float(wh[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"rectangle #{i} has a non-positive dimension")
            self.pieces.append(_Piece(i, w, h))

    # ------------------------------------------------------------------ #
    #  Necessary conditions
    # ------------------------------------------------------------------ #
    def feasibility_report(self) -> List[str]:
        """Reasons why *no* cover can possibly exist.  Empty list = unknown/ok.

        These are necessary conditions only; passing them does not prove that a
        cover exists.
        """
        W, H, problems = self.width, self.height, []

        # Pieces that cannot legally be used at all are worth nothing.
        usable = [p for p in self.pieces if self._orientations(p, (W, H))]
        if not usable:
            problems.append("no piece fits inside the big rectangle "
                            "(set allow_overhang=True to let pieces stick out)")
            return problems

        total = sum(p.area for p in usable)
        if total < W * H - EPS:
            problems.append(
                f"total area of the usable pieces ({total:g}) is smaller than "
                f"the target area ({W * H:g})")

        # A piece can cover two corners of the big rectangle only if it spans a
        # full side.  Otherwise each of the 4 corners needs its own piece.
        def spans_a_side(p: _Piece) -> bool:
            return any(w >= W - EPS or h >= H - EPS
                       for w, h, _ in self._orientations(p, (W, H)))

        if not any(spans_a_side(p) for p in usable) and len(usable) < 4:
            problems.append(
                "no piece spans a full side, so the 4 corners need 4 distinct "
                f"pieces, but only {len(usable)} are usable")

        return problems

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def solve(self, prune: bool = True) -> CoverResult:
        """Compute a covering.  Raises :class:`CoverError` on failure.

        ``prune=True`` runs a final pass that drops every placed piece whose
        removal still leaves the rectangle covered.  It costs one full
        verification per placed piece, so pass ``prune=False`` when there are
        many hundreds of pieces and speed matters more than tidiness.
        """
        problems = self.feasibility_report()
        if problems:
            raise CoverError("no cover exists: " + "; ".join(problems))

        candidates: List[CoverResult] = []

        # Pass 1: horizontal bands, swept top to bottom.
        rows = self._bands(self.width, self.height, self.pieces)
        if rows is not None:
            candidates.append(rows)

        # Pass 2: the same construction on the problem reflected across the
        # main diagonal, i.e. vertical bands swept left to right.  Reflecting
        # swaps every piece's dimensions as well, so the orientation labels
        # carry over unchanged and only the coordinates need swapping back.
        cols = self._bands(self.height, self.width,
                           [_Piece(p.index, p.h, p.w) for p in self.pieces])
        if cols is not None:
            candidates.append(self._reflect(cols))

        if not candidates:
            raise CoverError(
                "the shelf heuristic could not cover the rectangle with these "
                "pieces (the problem may still be solvable - it is NP-hard)")

        best = min(candidates, key=lambda r: (r.used_area, len(r.placements)))
        if prune:
            best = self._prune(best)
        return best

    def try_solve(self, prune: bool = True) -> Optional[CoverResult]:
        """Like :meth:`solve` but returns ``None`` instead of raising."""
        try:
            return self.solve(prune=prune)
        except CoverError:
            return None

    # ------------------------------------------------------------------ #
    #  Core construction
    # ------------------------------------------------------------------ #
    def _orientations(self, p: _Piece,
                      frame: Optional[Tuple[float, float]] = None
                      ) -> List[Tuple[float, float, int]]:
        """[(w, h, angle), ...] allowed for this piece.

        When ``frame`` is given and overhang is forbidden, orientations that do
        not fit inside the frame are dropped.
        """
        out = [(p.w, p.h, 0)]
        if self.allow_rotation and abs(p.w - p.h) > EPS:
            out.append((p.h, p.w, 90))
        if frame is not None and not self.allow_overhang:
            fw, fh = frame
            out = [o for o in out if o[0] <= fw + EPS and o[1] <= fh + EPS]
        return out

    def _bands(self, W: float, H: float, pieces: Sequence[_Piece]) -> Optional[CoverResult]:
        """Sweep top-to-bottom, filling one horizontal band at a time."""
        remaining = {p.index: p for p in pieces}
        placements: List[Placement] = []
        y = 0.0
        guard = 0

        while y < H - EPS:
            guard += 1
            if guard > 4 * len(pieces) + 16:          # cannot need more bands
                return None

            choice = self._best_band(W, H, H - y, remaining)
            if choice is None:
                return None
            gain, selected = choice
            if gain < EPS:
                return None

            band_h = min(o[2] for o in selected)
            # Keep the band inside the big rectangle: if it overshoots the
            # bottom edge, slide it up so its bottom edge lands exactly on H.
            # Sliding up is always safe - everything above y is covered already.
            top = y if y + band_h <= H + EPS else max(0.0, H - band_h)

            x = 0.0
            for idx, w, h, angle in selected:
                px = min(x, W - w) if not self.allow_overhang else x
                py = min(top, H - h) if not self.allow_overhang else top
                placements.append(Placement(index=idx,
                                            cx=px + w / 2.0,
                                            cy=py + h / 2.0,
                                            angle=angle,
                                            width=w,
                                            height=h))
                del remaining[idx]
                x = px + w

            y += gain

        return CoverResult(width=W, height=H,
                           placements=placements,
                           unused=sorted(remaining))

    def _best_band(self, W: float, H: float, remaining_height: float, remaining: dict
                   ) -> Optional[Tuple[float, List[Tuple[int, float, float, int]]]]:
        """Pick the band height and the pieces filling it.

        Returns ``(band_height, [(index, w, h, angle), ...])`` or ``None``.
        The scoring favours the band that gains the most height per unit of
        material consumed.
        """
        if not remaining:
            return None

        # Candidate band heights: every distinct height a piece can offer,
        # capped by what is left to cover.
        frame = (W, H)
        heights = set()
        for p in remaining.values():
            for _w, h, _a in self._orientations(p, frame):
                heights.add(min(h, remaining_height))
        heights.add(remaining_height)

        best_score = -1.0
        best: Optional[Tuple[float, List[Tuple[int, float, float, int]]]] = None

        for t in sorted(heights, reverse=True):
            if t < EPS:
                continue

            # For this band height, the widest orientation of every usable piece.
            usable = []
            for p in remaining.values():
                opts = [(w, h, a) for w, h, a in self._orientations(p, frame)
                        if h >= t - EPS]
                if not opts:
                    continue
                w, h, a = max(opts, key=lambda o: o[0])
                usable.append((p.index, w, h, a))
            if not usable:
                continue

            # Cheapest way to reach total width W is to take the pieces with the
            # smallest area-per-unit-width, which is exactly their height.
            usable.sort(key=lambda o: (o[2], -o[1]))
            chosen: List[Tuple[int, float, float, int]] = []
            acc = 0.0
            for item in usable:
                chosen.append(item)
                acc += item[1]
                if acc >= W - EPS:
                    break
            if acc < W - EPS:
                continue                                  # cannot span the width

            # Drop pieces that are not needed after all (largest area first).
            for item in sorted(chosen, key=lambda o: -o[1] * o[2]):
                if len(chosen) > 1 and acc - item[1] >= W - EPS:
                    chosen.remove(item)
                    acc -= item[1]

            band_h = min(o[2] for o in chosen)             # real covered height
            gain = min(band_h, remaining_height)
            area = sum(o[1] * o[2] for o in chosen)
            score = gain / area if area else 0.0

            if score > best_score + 1e-12:
                best_score = score
                best = (gain, chosen)

        return best

    # ------------------------------------------------------------------ #
    #  Post-processing
    # ------------------------------------------------------------------ #
    def _reflect(self, result: CoverResult) -> CoverResult:
        """Map a solution of the diagonally reflected problem back.

        The reflected problem was posed with every piece's dimensions already
        swapped, so an orientation labelled ``angle`` there denotes the same
        physical orientation here; only the coordinates and the footprint are
        exchanged.
        """
        flipped = [Placement(index=p.index,
                             cx=p.cy, cy=p.cx,
                             angle=p.angle,
                             width=p.height, height=p.width)
                   for p in result.placements]
        return CoverResult(width=self.width, height=self.height,
                           placements=flipped, unused=list(result.unused))

    def _prune(self, result: CoverResult) -> CoverResult:
        """Remove any placed piece that is fully redundant (biggest first)."""
        kept = list(result.placements)
        for cand in sorted(result.placements, key=lambda p: -(p.width * p.height)):
            trial = [p for p in kept if p is not cand]
            if not trial:
                continue
            probe = CoverResult(result.width, result.height, trial, [])
            if probe.verify():
                kept = trial
        unused = sorted(set(result.unused) | ({p.index for p in result.placements}
                                              - {p.index for p in kept}))
        return CoverResult(result.width, result.height, kept, unused)


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import random

    random.seed(7)
    W, H = 100.0, 60.0
    pieces = [(random.randint(8, 30), random.randint(5, 22)) for _ in range(40)]

    solver = RectangleCover(W, H, pieces)
    res = solver.solve()

    print(f"big rectangle : {W:g} x {H:g}  (area {W * H:g})")
    print(f"pieces given  : {len(pieces)}  (total area {sum(w * h for w, h in pieces):g})")
    print(f"pieces used   : {len(res.placements)}  (area {res.used_area:g})")
    print(f"overlap ratio : {res.overlap_ratio:.3f}")
    print(f"verified      : {res.verify()}")
    print()
    for p in sorted(res.placements, key=lambda q: q.index):
        print(f"  piece {p.index:2d}  centre=({p.cx:7.2f}, {p.cy:7.2f})  angle={p.angle:2d}")
