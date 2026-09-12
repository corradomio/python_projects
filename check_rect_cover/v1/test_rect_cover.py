"""Stress + edge-case tests for rect_cover.RectangleCover."""

import random
from rect_cover import RectangleCover, CoverError, CoverResult, Placement

FAIL = 0


def check(name, cond):
    global FAIL
    if not cond:
        FAIL += 1
        print(f"  FAIL  {name}")
    else:
        print(f"  ok    {name}")


def inside(res: CoverResult) -> bool:
    for p in res.placements:
        x0, y0, x1, y1 = p.bounds
        if x0 < -1e-9 or y0 < -1e-9 or x1 > res.width + 1e-9 or y1 > res.height + 1e-9:
            return False
    return True


def footprints_legal(res: CoverResult, rects, allow_rotation) -> bool:
    for p in res.placements:
        w, h = rects[p.index]
        if p.angle == 0:
            if abs(p.width - w) > 1e-9 or abs(p.height - h) > 1e-9:
                return False
        elif p.angle == 90:
            if not allow_rotation:
                return False
            if abs(p.width - h) > 1e-9 or abs(p.height - w) > 1e-9:
                return False
        else:
            return False
    return True


def no_duplicates(res: CoverResult) -> bool:
    idx = [p.index for p in res.placements]
    return len(idx) == len(set(idx))


print("== random stress, rotation allowed ==")
# random.seed(1)
solved = failed = 0
for trial in range(300):
    W = random.randint(20, 120)
    H = random.randint(20, 120)
    n = random.randint(5, 60)
    rects = [(random.randint(3, max(4, W // 2)), random.randint(3, max(4, H // 2)))
             for _ in range(n)]
    # make sure there is plenty of material
    while sum(w * h for w, h in rects) < 2.0 * W * H:
        rects.append((random.randint(3, max(4, W // 2)), random.randint(3, max(4, H // 2))))

    s = RectangleCover(W, H, rects)
    res = s.try_solve()
    if res is None:
        failed += 1
        continue
    solved += 1
    assert res.verify(), f"trial {trial}: coverage hole"
    assert inside(res), f"trial {trial}: piece outside bounds"
    assert footprints_legal(res, rects, True), f"trial {trial}: illegal footprint"
    assert no_duplicates(res), f"trial {trial}: piece used twice"
print(f"  solved {solved}/300, unsolved {failed}")
check("all random instances verified", failed == 0)

print("\n== random stress, rotation forbidden ==")
random.seed(2)
bad = 0
for trial in range(200):
    W, H = random.randint(20, 90), random.randint(20, 90)
    rects = [(random.randint(3, 25), random.randint(3, 25)) for _ in range(60)]
    while sum(w * h for w, h in rects) < 3.0 * W * H:
        rects.append((random.randint(3, 25), random.randint(3, 25)))
    s = RectangleCover(W, H, rects, allow_rotation=False)
    res = s.try_solve()
    if res is None:
        bad += 1
        continue
    assert res.verify() and inside(res)
    assert footprints_legal(res, rects, False), "rotation used although forbidden"
    assert all(p.angle == 0 for p in res.placements)
print(f"  unsolved {bad}/200")
check("no rotation leaked in", bad <= 4)

print("\n== exact tilings ==")
s = RectangleCover(4, 4, [(2, 2)] * 4)
r = s.solve()
check("2x2 squares tile 4x4 with 4 pieces", len(r.placements) == 4 and r.verify()
      and abs(r.overlap_ratio - 1.0) < 1e-9)

s = RectangleCover(10, 6, [(10, 6)])
r = s.solve()
check("single exact piece", len(r.placements) == 1 and r.verify()
      and r.placements[0].cx == 5 and r.placements[0].cy == 3)

s = RectangleCover(6, 10, [(10, 6)])
r = s.solve()
check("single piece needs rotation", len(r.placements) == 1 and r.verify()
      and r.placements[0].angle == 90)

print("\n== strips ==")
s = RectangleCover(100, 5, [(30, 5)] * 4)
r = s.solve()
check("1-D strip cover", r.verify() and inside(r))

s = RectangleCover(5, 100, [(30, 5)] * 4)
r = s.solve()
check("1-D vertical strip cover (needs the reflected pass)", r.verify() and inside(r))

print("\n== infeasible instances ==")
try:
    RectangleCover(1, 1, [(0.8, 0.8), (0.8, 0.8)]).solve()
    check("two 0.8 squares cannot cover the unit square", False)
except CoverError as e:
    check("two 0.8 squares cannot cover the unit square (corner argument)",
          "corners" in str(e))

try:
    RectangleCover(10, 10, [(3, 3)] * 5).solve()
    check("insufficient area rejected", False)
except CoverError as e:
    check("insufficient area rejected", "smaller than" in str(e))

try:
    RectangleCover(10, 10, [(20, 20)] * 5).solve()
    check("oversized pieces rejected when overhang is forbidden", False)
except CoverError as e:
    check("oversized pieces rejected when overhang is forbidden", "fits inside" in str(e))

r = RectangleCover(10, 10, [(20, 20)], allow_overhang=True).solve()
check("oversized piece accepted with allow_overhang", r.verify(inside_only=False))

print("\n== pruning ==")
rects = [(5, 5)] * 40
s = RectangleCover(10, 10, rects)
r = s.solve()
check("pruning finds the 4-piece cover of 10x10 by 5x5", len(r.placements) == 4)
check("unused pieces reported", len(r.unused) == 36)
check("centers() length matches", len(r.centers()) == len(r.placements))

print("\n== awkward aspect ratios ==")
random.seed(3)
odd = 0
for W, H in [(1, 1000), (1000, 1), (7, 999), (333, 5)]:
    lo = min(W, H)

    def gen():
        return (random.randint(1, max(1, min(9, lo))), random.randint(1, 9))

    rects = [gen() for _ in range(400)]
    while sum(w * h for w, h in rects) < 4 * W * H:
        rects.append(gen())
    res = RectangleCover(W, H, rects).try_solve()
    if res is None:
        odd += 1
        print(f"  unsolved {W}x{H}")
    else:
        assert res.verify() and inside(res), f"{W}x{H} bad"
check("extreme aspect ratios handled", odd == 0)

print("\n== float dimensions ==")
res = RectangleCover(10.5, 7.25, [(2.3, 1.9)] * 60).try_solve()
check("float geometry verified", res is not None and res.verify() and inside(res))

print()
print("ALL PASSED" if FAIL == 0 else f"{FAIL} FAILURES")
