import cv2

def list_cameras_by_index():
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:  # Attempt to read a frame
            break  # No more cameras found
        else:
            available_cameras.append(index)
            cap.release()  # Release the camera
        index += 1
    return available_cameras

camera_indices = list_cameras_by_index()
print(f"Available camera indices: {camera_indices}")