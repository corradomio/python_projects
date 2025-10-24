from math import sqrt
from mathx import sq
from waterdrop import waterdrop_info

# print(sq(0.7071067811865475))

print(waterdrop_info(R=1, theta=135, degree=True))

print(waterdrop_info(b=2, h=1, degree=True))

print(waterdrop_info(b=0, h=2, degree=True))

print(waterdrop_info(b=0, h=0, degree=True))

print(waterdrop_info(b=2, h=0.001, degree=True))

print(waterdrop_info(b=2, theta=0.001, degree=True))