import numpy as np

# ZYX == Yaw/Pitch/Roll

def euler_to_matrix(roll, pitch, yaw):
    # R = Rz * Ry * Rx

    # Rotation about x (roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    # Rotation about y (pitch)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    # Rotation about z (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combined Rotation Matrix
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


roll, pitch, yaw = 0.5, 0.1, 0.1


matrix = euler_to_matrix(roll, pitch, yaw)
print(matrix)


import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Define your Euler Angles (e.g., yaw, pitch, roll in radians)
# Example: 90 deg z-yaw, 0 pitch, 0 roll
euler_angles = [np.pi/2, 0, 0]

# 2. Convert to rotation object (specify axes order, e.g., 'zyx')
# 'zyx' means Euler angles are applied as rotation around Z, then new Y, then new X
r = R.from_euler('zyx', euler_angles)

# 3. Get the 3x3 rotation matrix
rotation_matrix = r.as_matrix()

print("Euler Angles (rad):", euler_angles)
print("Rotation Matrix:\n", rotation_matrix)

# Optionally: For degrees, set degrees=True
# r = R.from_euler('zyx', [90, 0, 0], degrees=True)
