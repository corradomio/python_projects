from math import pi, sin, cos
from random import random, uniform

import mitsuba as mi

from .mitsubax import SCALING, TRANSLATION, ROTATION
from .mitsubax import SHAPE_TYPES, SHAPE
from .mitsubax import ToWorld

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
# end


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
# Mitsuba primitives
# ---------------------------------------------------------------------------
#     "cube", "sphere", "rectangle", "disk", "cylinder",
#     "ellipsoids", "ellipsoidsmesh",
#     "bsplinecurve", "linearcurve", "sdfgrid",
#     "obj", "ply", "serialized",
#     "shapegroup", "instance",

def _shape(type: str, to_world=None, shape_id=None):
    assert isinstance(type, str)
    assert type in SHAPE_TYPES
    assert to_world is None or isinstance(to_world, mi.ScalarAffineTransform4f)
    assert shape_id is None or isinstance(shape_id, str)

    shape = {
        "type": type
    }
    if to_world is not None:
        shape["to_world"] = to_world
    if shape_id is not None:
        shape["id"] = shape_id
    return shape


def cube(to_world=None, shape_id=None) -> SHAPE:
    return _shape("cube", to_world, shape_id)


def sphere(to_world=None, shape_id=None) -> SHAPE:
    return _shape("sphere", to_world, shape_id)


def rectangle(to_world=None, shape_id=None) -> SHAPE:
    return _shape("rectangle", to_world, shape_id)


def disk(to_world=None, shape_id=None) -> SHAPE:
    return _shape("disk", to_world, shape_id)


def cylinder(to_world=None, shape_id=None) -> SHAPE:
    return _shape("cylinder", to_world, shape_id)


def shapegroup(components: dict, to_world=None, shape_id=None) -> SHAPE:
    for sub_id in components:
        shape = components[sub_id]
        shape["id"] = sub_id
    shape = {} | components | _shape("shapegroup", to_world, shape_id)
    return shape

# ---------------------------------------------------------------------------

def parallelepiped(sides: SCALING, translate: TRANSLATION, rotate: ROTATION=("z", 0), shape_id=None) -> SHAPE:
    t = ToWorld()\
        .scale(0.5, 0.5, 0.5)\
        .scale(*sides)\
        .rotate(value=rotate)\
        .translate(value=translate)\
        .get()
    return cube(t, shape_id)


def frame(side: float, width: float, height: float,
          rotate: ROTATION,
          translate: TRANSLATION) -> SHAPE:

    w2 = width/2
    h2 = height/2
    s2 = side/2
    f1 = parallelepiped((width, side, side), translate=[0,-h2,s2])
    f2 = parallelepiped((width, side, side), translate=[0, h2,s2])
    f3 = parallelepiped((side, height, side), translate=[-w2, 0,s2])
    f4 = parallelepiped((side, height, side), translate=[w2, 0,s2])



# ---------------------------------------------------------------------------
# OBJ Primitives
# ---------------------------------------------------------------------------

def ellipsoid(Nu: int, Nv: int, roughness:float = 0, suffix=""):
    """
    Ellipsoid created from top to bottom

    :param Nu: on XY plane
    :param Nv: on Z axis
    :return:
    """
    N = Nu*Nv
    du = twopi/Nu
    dv = pi/(Nv+1)

    points = []

    points.append([0,0,1])
    for v in range(1, Nv+1):
        phi = v*dv
        for u in range(Nu):
            theta = u*du

            rx = roughness*uniform(-1,1)
            ry = roughness*uniform(-1,1)
            rz = roughness*uniform(-1,1)

            x = _chop(sin(phi) * cos(theta)) + rx
            y = _chop(sin(phi) * sin(theta)) + ry
            z = _chop(cos(phi)) + rz

            points.append([x, y, z])
    # end
    points.append([0,0,-1])

    faces = []
    for v in range(Nv-1):
        v0 = 1 + v*Nu
        for u in range(Nu):
            v1 = v0 + u
            v2 = v0 + Nu + u
            v3 = v0 + Nu + (u + 1) % Nu
            v4 = v0 + (u + 1) % Nu

            faces.append([v1, v2, v3])
            faces.append([v3, v4, v1])
        # end
    # end

    # top face
    v0 = 0
    for v in range(Nu):
        v1 = 1 + v
        v2 = 1 + (v + 1) % Nu

        faces.append([v0,v1,v2])
    # end

    # bottom face
    v0 = N + 1
    L = N-Nu+1
    for v in range(Nu):
        v1 = L + v
        v2 = L + (v + 1) % Nu

        faces.append([v0, v2, v1])

    # Nl = Nu*(Nv-1)
    # faces.append([v for v in range(Nu)])
    # faces.append(reversed([Nl + v for v in range(Nu)]))

    if roughness > 0:
        _save_obj(f"objs/rough-ellipsoid-{Nu}x{Nv}{suffix}.obj", f"ellipsoid {Nu}x{Nv}", points, faces)
    else:
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


def rough_surface_2(Nw: int, Nh: int, rough: float):
    """

    :param Nw:
    :param Nh:
    :param rough:
    :return:
    """
    dx = 2/(Nw-1)
    dy = 2/(Nh-1)
    Nt = Nw*Nh          # total number of points
    Nl = Nw*(Nh-1)      # last line

    points = []

    # top surface
    for u in range(Nh):
        for v in range(Nw):
            y = u*dy - 1
            x = v*dx - 1
            z = rough*random()

            points.append([x, y, z])
        # end
    # end

    # bottom surface
    points += [
        [-1,-1,0],
        [-1, 1,0],
        [ 1, 1,0],
        [ 1,-1,0]
    ]

    bottom_indices = [u              for u in range(Nw)]
    top_indices = [Nl + u            for u in range(Nw)]
    left_indices  = [u*Nw            for u in range(Nh)]
    right_indices = [u*Nw + (Nw - 1) for u in range(Nh)]

    # print(top_indices)
    # print(bottom_indices)
    # print(left_indices)
    # print(right_indices)

    # ---------------------------------------------------

    faces = []
    for u in range(Nh-2):
        v0 = u*Nw
        for v in range(Nw-2):
            v1 = v0 + v
            v2 = v0 + v + Nw
            v3 = v0 + v + Nw + 1
            v4 = v0 + v + 1

            # if random() < .5:
            if (u+v)%2:
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
            else:
                faces.append([v1, v2, v4])
                faces.append([v4, v2, v3])
        # end
    # end

    base_face = [Nt+0, Nt+1, Nt+2, Nt+3]
    left_face = _reverse(left_indices) + [Nt + 0, Nt + 1]
    right_face = _reverse(right_indices) + [Nt + 3, Nt + 2]
    top_face = top_indices + [Nt+2, Nt+1]
    bottom_face = _reverse(bottom_indices) + [Nt + 0, Nt + 3]

    faces.append(base_face)
    faces.append(left_face)
    faces.append(right_face)
    faces.append(top_face)
    faces.append(bottom_face)

    _save_obj(f"objs/rough_surface-{Nw}x{Nh}.obj", f"rough surface {Nw}x{Nh}, {rough}", points, faces)
# end


def rough_surface_wh(w: float, h:float, rough: float):
    """
    Create a flat surface having 'w' width 'h' height and a roughness of 'rough'
    Then, everything is normalized in range [-1,1] (width, height) and [0,1] (roughness)

    Algorithm: the rectangle (w x h) is subdivided in (w/rough) and (h/rought)
    mini rectangles

    :param w:
    :param h:
    :param rough:
    :return:
    """
    rough_surface_2(int(w/rough), int(h/rough), rough)
# end
