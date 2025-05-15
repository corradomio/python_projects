import matplotlib.pyplot as plt
from math import pi, radians
from bezierx import CircleCurve, BezierCurve, DropletCurve


# cc = CircleCurve((0,0), 1, (0, pi/2))
# ccpts = cc.points()
#
# plt.plot(ccpts[:,0], ccpts[:,1])
# plt.show()

# bc = BezierCurve([
#     [0,0],
#     [-1,1.4],
#     [4,1.4],
#     [3,0]
# ],
# degree=3)
#
# bcpts = bc.points(30)
# print(bcpts)
# plt.plot(bcpts[:,0], bcpts[:,1])
#
# plt.ylim(0, 1.5)
# plt.show()
#
# bcpts = bc.points(30, nth=True)
# plt.plot(bcpts[:,0], bcpts[:,1])
#
# plt.ylim(0, 1.5)
# plt.show()

dc = DropletCurve(base=1,
        eccentricity=1,
        contact_angle=(radians(90), radians(90)))

# dcpts = dc.points(30)
# plt.plot(dcpts[:,0], dcpts[:,1])
# plt.ylim(0, 1.5)
# plt.show()

dcpts = dc.profile(10)
plt.plot(dcpts[:,0], dcpts[:,1])
# plt.ylim(0, 1.5)
plt.show()
