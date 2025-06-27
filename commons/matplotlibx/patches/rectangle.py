from typing import cast, Collection

import matplotlib.pyplot as plt
import matplotlib.patches as pch


class Rectangle(pch.Rectangle):
    def __init__(self, xy=None, width=None, height=None, *,
                 yx=None,
                 cxy=None,
                 cyx=None,
                 angle=0.0,
                 rotation_point='xy',
                 **kwargs):
        """
        Extension for matplotlib.patches.Rectangle:



        :param xy: (x,y) or (x1,y1,x2,y2)
        :param width:   width
        :param height:  height
        :param yx: (y,x) or (y1,x1,y2,x1)
        :param cxy: (x,y) of center
        :param cyx: (y,x) of center
        :param angle:
        :param rotation_point:
        :param kwargs:
        """
        if xy and width and height:
            pass
        elif xy and isinstance(xy, (list, tuple)) and len(xy) == 4:
            x1,y1,x2,y2 = xy
            xy = x1,y1
            width = x2-x1
            height = y2-y1
        elif yx and width and height:
            y, x = yx
            xy = x, y
        elif yx and  isinstance(yx, (list, tuple)) and len(yx) == 4:
            y1,x1,y2,x2 = cast(Collection, yx)
            xy = x1,y1
            width = x2-x1
            height = y2-y1
        elif cxy and width and height:
            cx, cy = cxy
            if isinstance(cx, float):
                xy = cx - width/2, cy - height/2
            else:
                xy = cx - width//2, cy - height//2
        elif cyx and width and height:
            cy, cx = cyx
            if isinstance(cx, float):
                xy = cx - width/2, cy - height/2
            else:
                xy = cx - width//2, cy - height//2

        super().__init__(xy=xy, width=width,height=height,
                         angle=angle, rotation_point=rotation_point,
                         **kwargs)
# end


def rectangle(xy=None, width=None, height=None, *,
              yx=None,
              cxy=None,
              cyx=None,
              angle=0.0, rotation_point='xy', **kwargs):
    ax = plt.gca()
    ax.add_patch(Rectangle(xy=xy, width=width, height=height,
                           yx=yx, cxy=cxy, cyx=cyx,
                           angle=angle, rotation_point=rotation_point, **kwargs))
# end
