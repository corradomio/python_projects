# Implements the conversion from different coordinate conventions:
#
#   xxyy    xmin,xmax,ymin,ymax
#   xyxy    xmin,ymin,xmax,ymax
#   xywh    xmin,ymin,width,height
#   cxcywh  center-x,center-y,width,height
#   ...n    normalized in range [0,1]
from typing import Literal

BBOX = tuple[float, float, float, float]

COORD_FORMATS = [
    "xxyy",  "xyxy",  "xywh",  "cxcywh",
    "xxyyn", "xyxyn", "xywhn", "cxcywhn",
]

FORMAT_TYPE = Literal["xxyy",  "xyxy",  "xywh",  "cxcywh",  "xxyyn", "xyxyn", "xywhn", "cxcywhn"]

#
# default format: "xyxy"
#

def _from(coords_: BBOX, w: float, h: float, from_format: FORMAT_TYPE)-> BBOX:
    coords = coords_
    if from_format.endswith("n"):
        if from_format.startswith("xxyy"):
            x1, x2, y1, y2 = coords
            coords = (x1*w, x2*w, y1*h, y2*h)
        else:
            x1, x2, x3, x4 = coords
            coords = (x1*w, x2*h, x3*w, x4*h)
        from_format = from_format[:-1]
    # end
    if from_format == "xxyy":
        x1, x2, y1,y2 = coords
    elif from_format == "xyxy":
        x1, y1, x2, y2 = coords
    elif from_format == "xywh":
        x1, y1, w, h = coords
        x2 = x1 + w
        y2 = y1 + h
    elif from_format == "cxcywh":
        cx, cy, w, h = coords
        w2 = w/2
        h2 = h/2
        x1 = cx - w2
        y1 = cy - h2
        x2 = cx + w2
        y2 = cy + h2
    else:
        raise ValueError("Unknown coordinate format: {}".format(from_format))
    return x1, y1, x2, y2


def _to(coords: BBOX,  w: float, h: float, to_format: FORMAT_TYPE)-> BBOX:
    x1, y1, x2, y2 = coords
    if to_format.endswith("n"):
        x1 /= w
        y1 /= h
        x2 /= w
        y2 /= h
        to_format = to_format[:-1]
    # end
    if to_format == "xxyy":
        return x1, x2, y1, y2
    elif to_format == "xyxy":
        return x1, y1, x2, y2
    elif to_format == "xywh":
        return x1, y1, x2-x1, y2-y1
    elif to_format == "cxcywh":
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h
    else:
        raise ValueError("Unknown coordinate format: {}".format(to_format))
    pass


def coordinate_convert(
    coords: BBOX, w: int, h: int,
    from_format: FORMAT_TYPE, to_format: FORMAT_TYPE) \
    -> BBOX:
    from_coords = _from(coords, w, h, from_format)
    to_coords = _to(from_coords, w, h, to_format)
    return to_coords
