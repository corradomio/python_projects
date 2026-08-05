from math import pi, sin, cos, atan, atan2, tan, radians, pow
import numpy as np


halfpi = pi/2
twopi = pi*2
eps = 1.e-10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sq(x: float) -> float:
    return x*x

def cube(x: float) -> float:
    return pow(x, 3.)


def croot(x: float) -> float:
    return pow(x, 0.3333333333)


def _chop(x):
    return x if x < -eps or x > eps else 0.


def _fwriteln(f, line):
    f.write(line)
    f.write('\n')


def _to_face(ilist):
    face = ""
    for i in ilist:
        if len(face) > 0:
            face += " "
        face += str(i+1)
    return face


def _save_obj(fname, header, vertices, faces):
    with open(fname, 'w') as ff:
        _fwriteln(ff, f"# {header}")
        for v in vertices:
            _fwriteln(ff, f"v {v[0]} {v[1]} {v[2]}")

        for s in faces:
            _fwriteln(ff, f"f {_to_face(s)}")


# ---------------------------------------------------------------------------
# DropletCurve
# ---------------------------------------------------------------------------

def droplet_radius(V: float, a: float) -> float:
    """
    Radius of the droplet given volume and contact angle

    :param V: volume
    :param a: contact angle
    :return:
    """
    return croot(3*V/pi/(2-3*cos(a)+cube(cos(a))))


def droplet_base(V: float, a: float) -> float:
    """
    Length of the contact base given volume and contact angle

    :param V: colume
    :param a: contact angle
    :return:
    """
    R = droplet_radius(V, a)
    b = halfpi - a
    return 2*R*cos(b)


def droplet_radius_angle_zmove(B: float, H: float) -> tuple[float, float, float]:
    """

    :param B:
    :param H:
    :return:
    """
    b4h = sq(B)+ 4*sq(H)
    R = b4h/(8*H)
    beta = atan2((sq(B) - 4*sq(H))/b4h, 4*B*H/b4h)
    alpha = halfpi - beta
    Z = R*sin(beta)

    return (R, beta * 180/pi, Z)
# end


class Droplet:
    def __init__(
            self,
            # base: float,
            volume: float,
            contact_angle: float,
            eccentricity: float = 1,
    ):
        """
        :param base: length of the contact solid/liquid
        :param contact_angle: (common | left, right) contact angle in the solid/liquid point
                in degrees
        :param eccentricity: ratio between short and long axes
        """
        contact_angle = radians(contact_angle)
        base = droplet_radius(volume, contact_angle)

        self.base = base
        self.eccentricity = eccentricity
        self.contact_angle: tuple[float, float] = (contact_angle, contact_angle)

        left_angle, right_angle = self.contact_angle

        left_angle = radians(left_angle)
        right_angle = radians(right_angle)

        if left_angle < right_angle:
            self.right_liquid_angle = atan(-self.eccentricity/tan(pi-right_angle))
            self.right_major_axis = self.base/2 / cos(self.right_liquid_angle)
            self.right_y = self.eccentricity * self.right_major_axis * sin(self.right_liquid_angle)
            self.right_x = self.right_major_axis *cos(self.right_liquid_angle)

            top = self.eccentricity*self.right_major_axis*(1-sin(self.right_liquid_angle))

            self.left_liquid_angle = atan(-self.eccentricity/tan(pi-left_angle))
            self.left_major_axis = top/(self.eccentricity*(1-sin(self.left_liquid_angle)))
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis *cos(self.left_liquid_angle)
        elif left_angle > right_angle:
            self.left_liquid_angle = atan(-self.eccentricity/tan(pi-left_angle))
            self.left_major_axis = self.base/2 / cos(self.left_liquid_angle)
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis *cos(self.left_liquid_angle)

            top = self.eccentricity*self.left_major_axis*(1-sin(self.left_liquid_angle))

            self.right_liquid_angle = atan(-self.eccentricity/tan(pi-right_angle))
            self.right_major_axis = top/(self.eccentricity*(1-sin(self.right_liquid_angle)))
            self.right_y = self.eccentricity * self.right_major_axis * sin(self.right_liquid_angle)
            self.right_x = self.right_major_axis *cos(self.right_liquid_angle)
            pass
        else:
            self.left_liquid_angle = atan(-self.eccentricity/tan(pi-left_angle))
            self.left_major_axis = self.base/2 / cos(self.left_liquid_angle)
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis *cos(self.left_liquid_angle)

            self.right_liquid_angle = self.left_liquid_angle
            self.right_major_axis = self.left_major_axis
            self.right_y = self.left_y
            self.right_x = self.right_major_axis *cos(self.right_liquid_angle)
            pass

        self._points = None
        return
    # end

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

    def points(self, Nu:int = 10) -> np.ndarray:
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

        self._points[ 0, 1] = 0
        self._points[-1, 1] = 0

        return self._points

    def profile(self, dy=0.01):
        xls = []
        xrs = []
        ys = []
        ymin, ymax = self.ylim()
        y = ymin
        while y < ymax:
            xl, xr = self.xat(y)
            xls.append(xl)
            xrs.append(xr)
            ys.append(y)
            y += dy
        # end
        # if y != ymax:
        #     xl, xr = self.xat(ymax)
        #     xls.append(xl)
        #     xrs.append(xr)
        #     ys.append(ymax)
        # # end

        n1 = len(ys)
        n2 = n1*2
        data = np.zeros((n2, 2))
        data[0:n1, 0] = xls
        data[n1:n2, 0] = list(reversed(xrs))
        data[0:n1,1] = ys
        data[n1:n2,1] = list(reversed(ys))
        return data
    # end

    def surface(self, Nu: int, Nv: int):
        self.points(4*Nu)

        ymin, ymax = self.ylim()
        ny = Nu
        dy = (ymax - ymin)/ny
        da = twopi/Nv

        profile = self.profile(dy)
        n: int = len(profile)//2
        ll = n-1
        rr = n+0

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

            for j in range(Nv):
                a = j*da
                xc = _chop(c + r*cos(a))
                yc = _chop(c + r*sin(a))
                zc = y

                xyz.append([xc,yc,zc])
                a += da
            # end
            surface.extend(xyz)
        # end
        # assert len(surface) == Nu*Nv
        points = np.array(surface)
        return surface
    # end

    # def plot(self, ax: Axes,
    #          border=False, fill=True, control_points=False, control_box=False,
    #          c='red', f='black', cp='green', cb='blue'):
    #     xy = self.points()
    #     if fill:
    #         ax.fill(xy[0], xy[1], c=f)
    #     if border:
    #         ax.plot(xy[0], xy[1], c=c)
    #     if control_points:
    #         ax.scatter(xy[0], xy[1], c=cp, s=10)
    # # end

    def save_obj(self, dir: str, Nu:int=10, Nv: int=20):
        points = self.surface(Nu, Nv)
        Np = len(points)

        faces = []
        for u in range(Nu - 1):
            v0 = u * Nv
            for v in range(Nv):
                w = (v + 1) % Nv
                v1 = v0 + v
                v2 = v0 + Nv + v
                v3 = v0 + Nv + w
                v4 = v0 + w

                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
            # end
        # end

        Nl = Nv * (Nu - 1)

        # top
        faces.append([v for v in range(Nv)])
        # base
        faces.append(reversed([Nl + v for v in range(Nv)]))

        ca_left, ca_right = self.contact_angle
        fname = f"droplet-{ca_left}x{ca_right}-{Nu}x{Nv}.obj"
        header = f"droplet: {ca_left}x{ca_right} {Nu}x{Nv}"
        _save_obj(f"{dir}/{fname}", header, points, faces)
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
