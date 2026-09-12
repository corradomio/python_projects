"""
rect_cover.py
=============

Cover a big rectangle with a collection of smaller rectangles, optionally
forcing a minimum overlap between neighbouring pieces.

The pieces may overlap each other, the whole big rectangle must end up covered,
and (by default) nothing is allowed to stick out of it.

Main entry point
----------------

    solver = RectangleCover(width=100, height=60,
                            rectangles=[(10, 8), (12, 5), ...],
                            overlap=0.25)
    result = solver.solve()
    for p in result.placements:
        print(p.index, p.cx, p.cy, p.angle)      # centre + orientation (0 or 90)

The overlap parameter
---------------------
``overlap`` is a number in [0, 1).  Two *neighbouring* pieces are guaranteed to
share a strip of at least ``overlap x min(extent_a, extent_b)``, measured along
the direction in which they are neighbours:

* pieces sitting side by side in the same band overlap horizontally by at least
  ``overlap x min(width_a, width_b)``;
* consecutive bands overlap vertically by at least
  ``overlap x min(height_a, height_b)``.

``overlap=0`` reproduces the plain edge-to-edge covering.  ``overlap=1`` is
rejected: it would mean each new piece never advances the frontier, so no
finite cover exists.  The definition is symmetric - the fraction is taken of
the *smaller* of the two neighbours, so the guarantee holds for both of them.

Note that forcing an overlap of ``r`` inflates the material needed by roughly
``1 / (1 - r)^2``; :meth:`RectangleCover.area_budget` reports that estimate.

Algorithm
---------
A *shelf* (horizontal band) construction.  The big rectangle is swept top to
bottom.  At each step the solver chooses a band height ``t`` and a subset of the
still-unused pieces such that every chosen piece is at least ``t`` tall (in the
orientation it is given) and their widths, chained together with the required
overlap, span the full width ``W``.  Flushed to the top of the band, those
pieces cover the strip completely.  The next band starts one overlap short of
the previous band's bottom edge, and the process repeats.

Among all candidate band heights the solver picks the one maximising
``height_gained / area_consumed``, i.e. the greedy that wastes the least
material per unit of progress.  The whole construction is run twice - once
normally and once on the diagonally reflected problem (vertical bands) - and
the cheaper of the two results is returned.

This is a heuristic: the exact problem (minimum number of rectangles covering a
rectangle) is NP-hard.  Failure of :meth:`solve` therefore means "this solver
could not find a cover", not "no cover exists" - except when one of the
necessary conditions checked by :meth:`feasibility_report` is violated, in
which case no cover exists at all.

Verification
------------
``CoverResult.verify()`` re-checks coverage from scratch by coordinate
compression, and ``CoverResult.check_overlap()`` re-checks that every
documented adjacency really honours the requested overlap.  Neither trusts the
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        Footprint *after* the rotation has been applied, so ``width`` is always
        the horizontal extent.
    band, slot : int
        Which band the piece belongs to, and its rank inside that band.
        Consecutive slots of a band, and consecutive bands, are the adjacencies
        on which the overlap guarantee is defined.
    """

    index: int
    cx: float
    cy: float
    angle: int
    width: float
    height: float
    band: int = 0
    slot: int = 0

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
    overlap: float = 0.0
    axis: str = "row"                       # "row" = horizontal bands
    strips: List[Tuple[float, float]] = field(default_factory=list)

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
        return [(p.cx, p.cy, p.angle)
                for p in sorted(self.placements, key=lambda q: q.index)]

    def by_band(self) -> Dict[int, List[Placement]]:
        """Placements grouped by band and ordered by slot."""
        groups: Dict[int, List[Placement]] = {}
        for p in self.placements:
            groups.setdefault(p.band, []).append(p)
        for v in groups.values():
            v.sort(key=lambda q: q.slot)
        return groups

    # -- validation -------------------------------------------------------- #
    def verify(self, inside_only: bool = True) -> bool:
        """Independently re-check that the big rectangle is fully covered.

        Uses coordinate compression: the union of the placed rectangles covers
        the target iff every elementary cell of the grid induced by all the
        rectangle edges is covered, which is tested at the cell centre.
        """
        W, H = self.width, self.height
        boxes = [p.bounds for p in self.placements]
        if not boxes:
            return False

        if inside_only:
            for (x0, y0, x1, y1) in boxes:
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

    def min_overlap(self) -> float:
        """Smallest relative overlap actually achieved over all adjacencies.

        Returns ``inf`` when there is no adjacency at all (a single piece).
        """
        worst = float("inf")
        along_x = (self.axis == "row")

        # neighbours inside a band
        for pieces in self.by_band().values():
            for a, b in zip(pieces, pieces[1:]):
                ax0, ay0, ax1, ay1 = a.bounds
                bx0, by0, bx1, by1 = b.bounds
                if along_x:
                    shared = min(ax1, bx1) - max(ax0, bx0)
                    ref = min(a.width, b.width)
                else:
                    shared = min(ay1, by1) - max(ay0, by0)
                    ref = min(a.height, b.height)
                if ref > EPS:
                    worst = min(worst, shared / ref)

        # neighbouring bands
        for (s0, e0), (s1, e1) in zip(self.strips, self.strips[1:]):
            shared = min(e0, e1) - max(s0, s1)
            ref = min(e0 - s0, e1 - s1)
            if ref > EPS:
                worst = min(worst, shared / ref)
        return worst

    def check_overlap(self, required: Optional[float] = None) -> bool:
        """True when every adjacency honours the requested overlap."""
        need = self.overlap if required is None else required
        if need <= 0:
            return True
        got = self.min_overlap()
        return got == float("inf") or got >= need - 1e-9

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
                f'fill="{colour}" fill-opacity="0.38" stroke="{colour}" stroke-width="1"/>'
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


# (index, width, height, angle) of a piece committed to a band
_Slot = Tuple[int, float, float, int]


class RectangleCover:
    """Cover a ``width`` x ``height`` rectangle with the given small rectangles.

    Parameters
    ----------
    width, height : float
        Dimensions of the big rectangle to be covered.
    rectangles : sequence of (w, h)
        The available pieces.  Not all of them need to be used.
    allow_rotation : bool, default True
        Whether a piece may be turned by 90 degrees.
    allow_overhang : bool, default False
        Whether pieces may stick out of the big rectangle.
    overlap : float in [0, 1), default 0.0
        Minimum overlap forced between neighbouring pieces, as a fraction of
        the smaller of the two neighbours' extents.  See the module docstring.
    """

    def __init__(self,
                 width: float,
                 height: float,
                 rectangles: Iterable[Sequence[float]],
                 allow_rotation: bool = True,
                 allow_overhang: bool = False,
                 overlap: float = 0.0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("the big rectangle must have positive dimensions")
        if not 0.0 <= overlap <= 1.0:
            raise ValueError("overlap must lie in [0, 1]")
        if overlap >= 1.0 - 1e-12:
            raise ValueError(
                "overlap=1 would make every new piece coincide with its "
                "neighbour, so no finite cover exists; use a value < 1")

        self.width = float(width)
        self.height = float(height)
        self.allow_rotation = bool(allow_rotation)
        self.allow_overhang = bool(allow_overhang)
        self.overlap = float(overlap)

        self.pieces: List[_Piece] = []
        for i, wh in enumerate(rectangles):
            w, h = float(wh[0]), float(wh[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"rectangle #{i} has a non-positive dimension")
            self.pieces.append(_Piece(i, w, h))

    # ------------------------------------------------------------------ #
    #  Budgeting and necessary conditions
    # ------------------------------------------------------------------ #
    def area_budget(self) -> float:
        """Rough amount of material a cover at this overlap will consume.

        Forcing an overlap ``r`` in both directions shrinks each piece's
        effective contribution to ``(1 - r)`` of each of its sides, so the
        material needed grows like ``W * H / (1 - r)^2``.  This is an estimate
        for planning, not a bound.
        """
        k = 1.0 - self.overlap
        return self.width * self.height / (k * k)

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
        removal still leaves the rectangle covered *and* keeps the overlap
        guarantee intact.  It costs one full verification per placed piece, so
        pass ``prune=False`` when there are many hundreds of pieces and speed
        matters more than tidiness.
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
            hint = ""
            if self.overlap > 0:
                hint = (f"; forcing an overlap of {self.overlap:g} needs roughly "
                        f"{self.area_budget():g} of material and you supplied "
                        f"{sum(p.area for p in self.pieces):g}")
            raise CoverError(
                "the shelf heuristic could not cover the rectangle with these "
                "pieces (the problem may still be solvable - it is NP-hard)" + hint)

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
    #  Geometry helpers
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

    def _chain(self, widths: Sequence[float]) -> List[float]:
        """Left edges of a run of pieces laid out with the forced overlap.

        Piece ``i+1`` starts one overlap short of where piece ``i`` ends, the
        overlap being ``self.overlap * min(w_i, w_{i+1})`` so that the fraction
        holds for both neighbours.
        """
        ov = self.overlap
        xs: List[float] = []
        front = 0.0
        prev: Optional[float] = None
        for w in widths:
            x = 0.0 if prev is None else max(0.0, front - ov * min(prev, w))
            xs.append(x)
            front = x + w
            prev = w
        return xs

    def _chain_extent(self, widths: Sequence[float]) -> float:
        """How far to the right a run of pieces reaches."""
        if not widths:
            return 0.0
        xs = self._chain(widths)
        return xs[-1] + widths[-1]

    # ------------------------------------------------------------------ #
    #  Core construction
    # ------------------------------------------------------------------ #
    def _bands(self, W: float, H: float,
               pieces: Sequence[_Piece]) -> Optional[CoverResult]:
        """Sweep top-to-bottom, filling one horizontal band at a time."""
        ov = self.overlap
        remaining = {p.index: p for p in pieces}
        placements: List[Placement] = []
        strips: List[Tuple[float, float]] = []

        covered = 0.0            # everything above this y is covered
        prev_h: Optional[float] = None
        guard = 0

        while covered < H - EPS:
            guard += 1
            if guard > 8 * len(pieces) + 32:      # cannot legitimately need more
                return None

            choice = self._best_band(W, H, H - covered, remaining, prev_h)
            if choice is None:
                return None
            gain, band_h, selected = choice
            if gain < EPS:
                return None

            # Top of this band: one overlap above the previous band's bottom.
            top = covered if prev_h is None else covered - ov * min(prev_h, band_h)
            top = max(0.0, top)
            # Keep the band inside: if it overshoots the bottom edge, slide it
            # up.  That is always safe - it only deepens the overlap with the
            # band above, and everything above `covered` is covered already.
            if not self.allow_overhang and top + band_h > H + EPS:
                top = max(0.0, H - band_h)

            band_index = len(strips)
            widths = [w for _i, w, _h, _a in selected]
            lefts = self._chain(widths)

            for slot, ((idx, w, h, angle), x) in enumerate(zip(selected, lefts)):
                px, py = x, top
                if not self.allow_overhang:
                    # Pulling a piece back inside only increases its overlap
                    # with the neighbour before it, so the guarantee survives.
                    px = max(0.0, min(px, W - w))
                    py = max(0.0, min(py, H - h))
                placements.append(Placement(index=idx,
                                            cx=px + w / 2.0, cy=py + h / 2.0,
                                            angle=angle, width=w, height=h,
                                            band=band_index, slot=slot))
                del remaining[idx]

            strips.append((top, top + band_h))
            covered = top + band_h
            prev_h = band_h

        return CoverResult(width=W, height=H,
                           placements=placements,
                           unused=sorted(remaining),
                           overlap=ov, axis="row", strips=strips)

    def _best_band(self, W: float, H: float, remaining_height: float,
                   remaining: Dict[int, _Piece], prev_h: Optional[float]
                   ) -> Optional[Tuple[float, float, List[_Slot]]]:
        """Pick the band height and the pieces filling it.

        Returns ``(gain, band_height, [(index, w, h, angle), ...])`` or
        ``None``.  ``gain`` is how much further down the sweep gets, which is
        the band height minus the overlap with the band above.  The scoring
        favours the band gaining the most height per unit of material spent.
        """
        if not remaining:
            return None

        ov = self.overlap
        frame = (W, H)

        # Candidate band heights: every distinct height a piece can offer.
        heights = set()
        for p in remaining.values():
            for _w, h, _a in self._orientations(p, frame):
                heights.add(h)
        if not heights:
            return None

        best_score = -1.0
        best: Optional[Tuple[float, float, List[_Slot]]] = None

        for t in sorted(heights, reverse=True):
            if t < EPS:
                continue

            # For this band height, the widest orientation of every usable piece.
            usable: List[_Slot] = []
            for p in remaining.values():
                opts = [(w, h, a) for w, h, a in self._orientations(p, frame)
                        if h >= t - EPS]
                if not opts:
                    continue
                w, h, a = max(opts, key=lambda o: o[0])
                usable.append((p.index, w, h, a))
            if not usable:
                continue

            # Cheapest way to span the width is to take the pieces with the
            # smallest area-per-unit-width, which is exactly their height.
            usable.sort(key=lambda o: (o[2], -o[1]))
            chosen: List[_Slot] = []
            for item in usable:
                chosen.append(item)
                if self._chain_extent([c[1] for c in chosen]) >= W - EPS:
                    break
            if self._chain_extent([c[1] for c in chosen]) < W - EPS:
                continue                                  # cannot span the width

            # Drop pieces that turn out to be unnecessary (largest area first).
            for item in sorted(chosen, key=lambda o: -o[1] * o[2]):
                if len(chosen) <= 1:
                    break
                trial = [c for c in chosen if c is not item]
                if self._chain_extent([c[1] for c in trial]) >= W - EPS:
                    chosen = trial

            band_h = min(o[2] for o in chosen)             # real covered height
            step = band_h if prev_h is None else band_h - ov * min(prev_h, band_h)
            gain = min(step, remaining_height)
            area = sum(o[1] * o[2] for o in chosen)
            score = gain / area if area else 0.0

            if score > best_score + 1e-12:
                best_score = score
                best = (gain, band_h, chosen)

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
                             width=p.height, height=p.width,
                             band=p.band, slot=p.slot)
                   for p in result.placements]
        return CoverResult(width=self.width, height=self.height,
                           placements=flipped, unused=list(result.unused),
                           overlap=result.overlap, axis="column",
                           strips=list(result.strips))

    def _prune(self, result: CoverResult) -> CoverResult:
        """Remove every placed piece that is redundant, biggest first.

        A piece may only go if the rectangle stays covered *and* the overlap
        guarantee still holds for the adjacencies that close up behind it.
        """
        kept = list(result.placements)
        counts: Dict[int, int] = {}
        for p in kept:
            counts[p.band] = counts.get(p.band, 0) + 1

        for cand in sorted(result.placements, key=lambda p: -(p.width * p.height)):
            if counts.get(cand.band, 0) <= 1:
                continue                       # never empty a band
            trial = [p for p in kept if p is not cand]
            probe = CoverResult(result.width, result.height, trial, [],
                                result.overlap, result.axis, result.strips)
            if probe.verify() and probe.check_overlap():
                kept = trial
                counts[cand.band] -= 1

        # Renumber slots so that "consecutive slots" stays meaningful.
        renumbered: List[Placement] = []
        for band, pieces in sorted(CoverResult(result.width, result.height, kept, [],
                                               result.overlap, result.axis,
                                               result.strips).by_band().items()):
            for slot, p in enumerate(pieces):
                renumbered.append(Placement(p.index, p.cx, p.cy, p.angle,
                                            p.width, p.height, band, slot))

        dropped = {p.index for p in result.placements} - {p.index for p in renumbered}
        return CoverResult(result.width, result.height, renumbered,
                           sorted(set(result.unused) | dropped),
                           result.overlap, result.axis, list(result.strips))


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import random

    W, H = 100.0, 60.0

    for ov in (0.0, 0.2, 0.4):
        random.seed(7)
        pieces = [(random.randint(8, 30), random.randint(5, 22)) for _ in range(40)]
        while sum(w * h for w, h in pieces) < 1.4 * W * H / (1 - ov) ** 2:
            pieces.append((random.randint(8, 30), random.randint(5, 22)))

        solver = RectangleCover(W, H, pieces, overlap=ov)
        res = solver.solve()
        print(f"overlap={ov:.1f}  pieces available={len(pieces):3d}  used={len(res.placements):3d}  "
              f"material={res.overlap_ratio:.3f}x  min overlap achieved={res.min_overlap():.3f}  "
              f"covered={res.verify()}  overlap ok={res.check_overlap()}")
