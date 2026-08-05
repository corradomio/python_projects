from math import pi, sin, cos
from random import random


twopi = 2*pi
halfpi = 0.5*pi
eps = 1.e-10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _chop(x):
    return x if x < -eps or x > eps else 0.


def _deg(a):
    return a*180/pi


def _reverse(l : list | tuple) -> list:
    return list(reversed(l))


# ---------------------------------------------------------------------------
# save_obj
# ---------------------------------------------------------------------------

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
# end


def _save_obj(fname, header, vertices, faces):
    with open(fname, 'w') as ff:
        _fwriteln(ff, f"# {header}")
        for v in vertices:
            _fwriteln(ff, f"v {v[0]} {v[1]} {v[2]}")

        for s in faces:
            _fwriteln(ff, f"f {_to_face(s)}")
# end



# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def ellipsoid(Nu: int, Nv: int):
    """
    Ellipsoid created from top to bottom

    :param Nu: on XY plane
    :param Nv: on Z axis
    :return:
    """
    du = twopi/Nu
    dv = pi/(Nv+1)

    points = []

    for v in range(1, Nv+1):
        phi = v*dv
        for u in range(Nu):
            theta = u*du

            x = _chop(sin(phi) * cos(theta))
            y = _chop(sin(phi) * sin(theta))
            z = _chop(cos(phi))

            points.append([x, y, z])
    # end

    faces = []
    for v in range(Nv-1):
        v0 = v*Nu
        for u in range(Nu):
            v1 = v0 + u
            v2 = v0 + Nu + u
            v3 = v0 + Nu + (u + 1) % Nu
            v4 = v0 + (u + 1) % Nu

            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
        # end
    # end

    Nl = Nu*(Nv-1)

    faces.append([v for v in range(Nu)])
    faces.append(reversed([Nl + v for v in range(Nu)]))

    _save_obj(f"objs/ellipsoid-{Nu}x{Nv}.obj", f"ellipsoid {Nu}x{Nv}", points, faces)
# end


def rough_surface(Np: int, rough: float):
    dp = 2/(Np-1)
    N2 = Np*Np
    N1 = Np*(Np-1)

    points = []

    # top surface
    for v in range(Np):
        for u in range(Np):
            y = u*dp - 1
            x = v*dp - 1
            z = rough*random()

            points.append([x, y, z])
        # end
    # end

    # bottom surface
    # for v in range(Np):
    #     for u in range(Np):
    #         x = u*dp
    #         y = v*dp
    #         z = 0
    #
    #         points.append([x, y, z])
    #     # end
    # # end
    points += [
        [-1,-1,0],
        [-1,1,0],
        [1,-1,0],
        [1,1,0]
    ]

    top_indices = [u for u in range(Np)]
    bottom_indices = [N1 + u for u in range(Np)]
    left_indices = [u*Np for u in range(Np)]
    right_indices = [((u+1)*Np - 1) for u in range(Np)]

    # print(top_indices)
    # print(bottom_indices)
    # print(left_indices)
    # print(right_indices)

    # ---------------------------------------------------

    faces = []
    for v in range(Np-1):
        v0 = v*Np
        for u in range(Np-1):
            v1 = v0 + u
            v2 = v0 + Np + u
            v3 = v0 + Np + (u + 1) % Np
            v4 = v0 + (u + 1) % Np

            # if random() < .5:
            if (u+v)%2:
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
            else:
                faces.append([v1, v2, v4])
                faces.append([v4, v2, v3])
        # end
    # end

    base_face = [N2+0, N2+1, N2+3, N2+2]
    left_face = _reverse(left_indices) + [N2 + 0, N2 + 2]
    bottom_face = _reverse(bottom_indices) + [N2 + 2, N2 + 3]
    right_face = right_indices + [N2+3, N2+1]
    top_face = top_indices + [N2+1, N2+0]

    faces.append(base_face)
    faces.append(left_face)
    faces.append(bottom_face)
    faces.append(right_face)
    faces.append(top_face)

    _save_obj(f"objs/rough_surface-{Np}x{Np}.obj", f"rough surface {Np}x{Np}, {rough}", points, faces)
# end



def main():
    # ellipsoid(5, 3)
    # ellipsoid(20, 15)
    # ellipsoid(40, 30)
    ellipsoid(80, 60)

    # rough_surface(2, 0)
    # rough_surface(10, 1)
    # rough_surface(20, 1)
    # rough_surface(50, 1)
    # rough_surface(90, 1)

    pass

if __name__ == "__main__":
    main()
