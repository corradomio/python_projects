from math import pi, sin, cos, atan, tan, radians
import numpy as np
from matplotlib.pylab import Axes


halfpi = pi/2
twopi = pi*2
eps = 1.e-10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def chop(x):
    return x if x < -eps or x > eps else 0.


def fwriteln(f, line):
    f.write(line)
    f.write('\n')


def to_face(ilist):
    face = ""
    for i in ilist:
        if len(face) > 0:
            face += " "
        face += str(i+1)
    return face
# end


def save_obj(fname, header, vertices, faces):
    with open(fname, 'w') as ff:
        fwriteln(ff, f"# {header}")
        for v in vertices:
            fwriteln(ff, f"v {v[0]} {v[1]} {v[2]}")

        for s in faces:
            fwriteln(ff, f"f {to_face(s)}")
# end


# ---------------------------------------------------------------------------
# DropletCurve
# ---------------------------------------------------------------------------

class DropletCurve:
    def __init__(
            self,
            base: float,
            contact_angle: float | tuple[float, float],
            eccentricity: float = 1.,
    ):
        self.base = base
        self.eccentricity = eccentricity

        self.contact_angle: tuple[float, float] = (
            (contact_angle, contact_angle) if isinstance(contact_angle, (float, int)) else contact_angle
        )

        left_angle, right_angle = self.contact_angle
        left_angle = radians(left_angle)
        right_angle = radians(right_angle)

        if left_angle < right_angle:
            self.right_liquid_angle = atan(-self.eccentricity / tan(pi - right_angle))
            self.right_major_axis = self.base / 2 / cos(self.right_liquid_angle)
            self.right_y = self.eccentricity * self.right_major_axis * sin(self.right_liquid_angle)
            self.right_x = self.right_major_axis * cos(self.right_liquid_angle)

            top = self.eccentricity * self.right_major_axis * (1 - sin(self.right_liquid_angle))

            self.left_liquid_angle = atan(-self.eccentricity / tan(pi - left_angle))
            self.left_major_axis = top / (self.eccentricity * (1 - sin(self.left_liquid_angle)))
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis * cos(self.left_liquid_angle)
        elif left_angle > right_angle:
            self.left_liquid_angle = atan(-self.eccentricity / tan(pi - left_angle))
            self.left_major_axis = self.base / 2 / cos(self.left_liquid_angle)
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis * cos(self.left_liquid_angle)

            top = self.eccentricity * self.left_major_axis * (1 - sin(self.left_liquid_angle))

            self.right_liquid_angle = atan(-self.eccentricity / tan(pi - right_angle))
            self.right_major_axis = top / (self.eccentricity * (1 - sin(self.right_liquid_angle)))
            self.right_y = self.eccentricity * self.right_major_axis * sin(self.right_liquid_angle)
            self.right_x = self.right_major_axis * cos(self.right_liquid_angle)
            pass
        else:
            self.left_liquid_angle = atan(-self.eccentricity / tan(pi - left_angle))
            self.left_major_axis = self.base / 2 / cos(self.left_liquid_angle)
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis * cos(self.left_liquid_angle)

            self.right_liquid_angle = self.left_liquid_angle
            self.right_major_axis = self.left_major_axis
            self.right_y = self.left_y
            self.right_x = self.right_major_axis * cos(self.right_liquid_angle)
            pass

        self._points = None
        return

    def xlim(self):
        xmin, xmax = 1000, -1000
        points = self.points()
        n = len(points)

        for i in range(n):
            x = points[i,0]
            if x < xmin: xmin = x
            if x > xmax: xmax = x
        return (float(xmin), float(xmax))

    def ylim(self):
        ymin, ymax = 1000, -1000
        points = self.points()
        n = len(points)

        for i in range(n):
            y = points[i,1]
            if y < ymin: ymin = y
            if y > ymax: ymax = y
        return (float(ymin), float(ymax))

    def xbase(self):
        return self.left_x, self.right_x

    def xat(self, y:float) -> tuple[float, float]:
        xleft = 1000
        xright = -1000
        points = self.points()
        n = len(points)
        l = 0
        for i in range(n-1):
            y0 = points[i+0, 1]
            y1 = points[i+1, 1]
            if y0 <= y < y1:
                x0 = points[i+0, 0]
                x1 = points[i+1, 0]
                xleft = x0 + (x1 - x0)*(y - y0)/(y1 - y0)
                l = i
                break
        for i in range(l+1, n-1):
            y0 = points[i+0, 1]
            y1 = points[i+1, 1]
            if y0 > y >= y1:
                x0 = points[i+0, 0]
                x1 = points[i+1, 0]
                xright = x0 + (x1 - x0)*(y - y0)/(y1 - y0)
                break

        if xright == -1000: xright = xleft
        return float(xleft), float(xright)
    # end

    def points(self, Nu: int=10) -> np.ndarray:
        if self._points is not None:
            return self._points

        left_angles: np.ndarray = np.linspace(pi-self.left_liquid_angle, radians(90), Nu//2)
        right_angles: np.ndarray = np.linspace(radians(90), self.right_liquid_angle, Nu//2)

        left_x = self.left_major_axis*np.cos(left_angles)
        left_y = self.eccentricity*self.left_major_axis*np.sin(left_angles)-self.left_y

        right_x = self.right_major_axis*np.cos(right_angles)
        right_y = self.eccentricity*self.right_major_axis*np.sin(right_angles)-self.right_y

        x = np.concatenate((left_x, right_x))
        y = np.concatenate((left_y, right_y))

        self._points = np.array([x, y]).T

        self._points[0, 1] = 0
        self._points[-1,1] = 0

        return self._points
    # end

    def profile(self, Nu: int=10) -> np.ndarray:
        xls = []
        xrs = []
        ys = []
        ymin, ymax = self.ylim()
        y = ymin
        dy = (ymax-ymin)/Nu
        while y < ymax:
            xl, xr = self.xat(y)
            xls.append(xl)
            xrs.append(xr)
            ys.append(y)
            y += dy
        # end

        n1 = len(ys)
        n2 = n1*2-1
        data = np.zeros((n2, 2))
        data[0:n1, 0] = xls
        data[n1-1:n2, 0] = list(reversed(xrs))
        data[0:n1,1] = ys
        data[n1-1:n2,1] = list(reversed(ys))

        return data
    # end

    # def surface(self, Nu: int=10, Nv: int=20):
    #
    #     profile = self.profile(Nu)
    #     n: int = len(profile)//2
    #     da = twopi/Nv
    #
    #     surface = []
    #
    #     for i in range(n):
    #         xl = profile[i+0+0,0]
    #         xr = profile[n-i-1, 0]
    #         y =  profile[i, 1]
    #         r = (xr-xl)/2
    #         c = (xr+xl)/2
    #
    #         xyz = []
    #
    #         a = 0
    #         while a < twopi:
    #             xc = c + r*cos(a)
    #             yc = c + r*sin(a)
    #             zc = y
    #
    #             xyz.append([xc,yc,zc])
    #
    #             a += da
    #         # end
    #         surface.append(xyz)
    #     # end
    #     return surface
    # # end

    def surface(self, Nu: int, Nv: int):

        profile = self.profile(Nu)
        np = len(profile)
        n: int = np//2
        ll = n-1
        rr = n+1
        da = twopi/Nu

        surface = []

        for i in range(n):
            xl = profile[ll-i, 0]
            xr = profile[rr+i, 0]
            yl = profile[ll-i, 1]
            yr = profile[rr+i, 1]
            assert yl == yr
            r = (xr-xl)/2
            c = (xr+xl)/2
            y = yl

            xyz = []

            for j in range(Nu):
                a = j*da
                xc = chop(c + r*cos(a))
                yc = chop(c + r*sin(a))
                zc = y

                xyz.append([xc,yc,zc])
                a += da
            # end
            surface.extend(xyz)
        # end
        # assert len(surface) == Nu*Nv
        return surface
    # end

    def plot(self, ax: Axes,
             border=False, fill=True, control_points=False, control_box=False,
             c='red', f='black', cp='green', cb='blue'):
        xy = self.points()
        if fill:
            ax.fill(xy[0], xy[1], c=f)
        if border:
            ax.plot(xy[0], xy[1], c=c)
        if control_points:
            ax.scatter(xy[0], xy[1], c=cp, s=10)
    # end

    def save_obj(self, dir: str, Nu:int=10, Nv: int=20):
        points = self.surface(Nu, Nv)

        faces = []
        for v in range(Nv - 1):
            v0 = v * Nu
            for u in range(Nu):
                v1 = v0 + u
                v2 = v0 + Nu + u
                v3 = v0 + Nu + (u + 1) % Nu
                v4 = v0 + (u + 1) % Nu

                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
            # end
        # end

        Nl = Nu * (Nv - 1)

        faces.append([v for v in range(Nu)])
        faces.append(reversed([Nl + v for v in range(Nu)]))

        ca_left, ca_right = self.contact_angle
        save_obj(f"{dir}/liquid_drop-{ca_left}x{ca_right}-{Nu}x{Nv}.obj", f"liquid drop: {ca_left}x{ca_right} {Nu}x{Nv}",
                 points, faces)
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
