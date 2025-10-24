from math import pi, sqrt
from waterdrop import droplet_radius_angle_zmove


R, beta, alpha, Z = droplet_radius_angle_zmove(B=0,H=1)
print(R, beta*180/pi, alpha*180/pi, Z)