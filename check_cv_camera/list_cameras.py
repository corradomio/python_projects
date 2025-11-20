import cv2_enumerate_cameras

# Enumerate cameras
camera_list = cv2_enumerate_cameras.enumerate_cameras()

# Print information for each camera
for camera_info in camera_list:
    print(f"Index: {camera_info.index}")
    print(f"Name: {camera_info.name}")
    print(f"Path: {camera_info.path}")
    print(f"Backend: {camera_info.backend}")
    print("-" * 20)
