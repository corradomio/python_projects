import numpy as np
from typing import Optional
from math import pi, sin, cos, pow, atan, tan, radians
from matplotlib.pylab import Axes

halfpi = pi/2
twopi = pi*2

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def sq(x): return x*x
def cu(x): return x*x*x

def bin(n, k):
    f = 1
    for i in range(k):
        f = f*(n-i)//(i+1)
    return f

def ipow(x, e):
    p = 1.
    for i in range(e):
        p *= x
    return p


# ---------------------------------------------------------------------------
# CircleCurve
# ---------------------------------------------------------------------------

class CircleCurve:
    def __init__(self, center: tuple[float, float], radius: float, angles: Optional[tuple[float, float]]):
        self.center = center
        self.radius = radius
        self.angles = (0, 2*pi) if angles is None else angles
    # end

    def points(self, Nu: int=10) -> np.ndarray:
        sa, ea = self.angles
        da = (ea-sa)/(Nu-1)
        xc, yc = self.center
        r = self.radius

        points = []
        a = sa
        while a <= ea:
            x = xc + r*cos(a)
            y = yc + r*sin(a)
            points.append((x, y))
            a += da
        # end
        return np.array(points)
    # end
# end



# ---------------------------------------------------------------------------
# BezierCurve
# ---------------------------------------------------------------------------

#
# https://en.wikipedia.org/wiki/B%C3%A9zier_curve
#
ListCoords = list[tuple[float, float]]


class BezierCurve:
    def __init__(self, nodes, degree: Optional[int]):
        self.nodes = nodes
        self.degree = len(nodes)-1 if degree is None else degree
        assert degree <= len(nodes) - 1
    # end

    def points(self, Nu: int=10, nth=False) -> np.ndarray:
        points = [self.nodes[0]]
        n = len(self.nodes)-1
        i = 0
        while i < n:
            d = min(self.degree, n-i)
            if nth:
                pts = self._ndeg(Nu, i, d)
                i += d
            elif d == 1:
                pts = self._one(Nu, i)
                i += 1
            elif d == 2:
                pts = self._two(Nu, i)
                i += 2
            elif d == 3:
                pts = self._three(Nu, i)
                i += 3
            else:
                pts = self._ndeg(Nu, i, d)
                i += d
            points.extend(pts)
        # end
        # points.append(self.nodes[-1])
        return np.array(points)
    # end

    def _one(self, Nu: int, i: int) -> ListCoords:
        points = []
        dt = 1./(Nu-1)
        t = dt
        x0, y0 = self.nodes[i+0]
        x1, y1 = self.nodes[i+1]
        for j in range(1, Nu):
            xp = (1-t)*x0 + t*x1
            yp = (1-t)*y0 + t*y1

            points.append((xp,yp))

            t += dt
        # end
        return points
    # end

    def _two(self, Nu: int, i: int) -> ListCoords:
        points = []
        dt = 1. / (Nu - 1)
        t = dt
        x0, y0 = self.nodes[i + 0]
        x1, y1 = self.nodes[i + 1]
        x2, y2 = self.nodes[i + 2]
        for j in range(1, Nu):
            xp = sq(1-t)*x0 + 2*(1-t)*t*x1 + sq(t)*x2
            yp = sq(1-t)*y0 + 2*(1-t)*t*y1 + sq(t)*y2

            points.append((xp, yp))

            t += dt
        # end
        return points
    # end

    def _three(self, Nu: int, i: int) -> ListCoords:
        points = []
        dt = 1. / (Nu - 1)
        t = dt
        x0, y0 = self.nodes[i + 0]
        x1, y1 = self.nodes[i + 1]
        x2, y2 = self.nodes[i + 2]
        x3, y3 = self.nodes[i + 3]
        for j in range(1, Nu):
            xp = cu(1 - t)*x0 + 3*sq(1 - t)*t*x1 + 3*(1-t)*sq(t)*x2 + cu(t)*x3
            yp = cu(1 - t)*y0 + 3*sq(1 - t)*t*y1 + 3*(1-t)*sq(t)*y2 + cu(t)*y3

            points.append((xp, yp))

            t += dt
        # end
        return points
    # end

    def _ndeg(self, Nu: int, i: int, n: int) -> ListCoords:
        points = []
        dt = 1. / (Nu - 1)
        t = dt
        for j in range(1, Nu):
            xp, yp = 0., 0.
            for k in range(n+1):
                xc, yc = self.nodes[i+k]
                xp += bin(n,k)*pow(1-t, n-k)*pow(t,k)*xc
                yp += bin(n,k)*pow(1-t, n-k)*pow(t,k)*yc
            # end

            points.append((xp, yp))

            t += dt
        # end
        return points
    # end
# end
