from typing import Any

from .mitsubax import register_procedural_shape, VECTOR, to_axis
from .primitives import parallelepiped, shapegroup


def cage(
    size: VECTOR,
    hbars: int, vbars: int,
    hbars_sides: str,
    frame_side: float=0.05,
    bar_diameter: float=0.01,
    hbar_sides = "",
    vbar_sides = ""
) -> dict[str, Any]:
    sx, sy, sz = size
    fs = frame_side
    
    sx2 = sx /2
    sy2 = sy /2
    sz2 = sz /2
    fs2 = fs /2

    shape = shapegroup(dict(

        bl = parallelepiped([fs, sy, fs], translate=[-sx2,0, fs2]),
        br = parallelepiped([fs, sy, fs], translate=[+sx2,0, fs2]),
        bf = parallelepiped([sx, fs, fs], translate=[0, -sy2, fs2]),
        bb = parallelepiped([sx, fs, fs], translate=[0, +sy2, fs2]),

        tl = parallelepiped([fs, sy, fs], translate=[-sx2,0, sz-fs2]),
        tr = parallelepiped([fs, sy, fs], translate=[+sx2,0, sz-fs2]),
        tf = parallelepiped([sx, fs, fs], translate=[0, -sy2, sz-fs2]),
        tb = parallelepiped([sx, fs, fs], translate=[0, +sy2, sz-fs2]),

        e1 = parallelepiped([fs, fs, sz], translate=[-sx2,-sy2,sz2]),
        e2 = parallelepiped([fs, fs, sz], translate=[ sx2,-sy2,sz2]),
        e3 = parallelepiped([fs, fs, sz], translate=[ sx2, sy2,sz2]),
        e4 = parallelepiped([fs, fs, sz], translate=[-sx2, sy2,sz2])
    ))


    return shape


register_procedural_shape("cage", cage)
