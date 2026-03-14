import cv2
import cv2 as cv
from vidgear.gears import CamGear
from stdlib.tprint import tprint

ips = ["10.248.37.111", "10.248.37.108", "10.248.37.12", "10.248.37.100"]
username = "admin"
password = "password1234"
all_rtsp_urls = [f"rtsp://{username}:{password}@{ip}/Streaming/Channels/101?tcp" for ip in ips]


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 10

OPTIONS = {
    "CAP_PROP_FRAME_WIDTH": CAMERA_WIDTH,
    "CAP_PROP_FRAME_HEIGHT": CAMERA_HEIGHT,
    "CAP_PROP_FPS": CAMERA_FPS # Optional: set FPS too
}

cv2.namedWindow("preview")

# Open live video stream on webcam at first index(i.e. 0) device
# stream = CamGear(source=0, **OPTIONS).start()
stream = CamGear(source=all_rtsp_urls[0], **OPTIONS).start()

# loop over
count = 0
while True:
    # read frames from stream
    frame = stream.read()

    # check for frame if None-type
    if frame is None:
        tprint("No frames available")
        break

    # {do something with the frame here}
    frame = cv.flip(frame, 1)
    cv2.imshow("preview", frame)

    # check for 'q' key if pressed

    count += 1
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # exit on ESC
        break
    tprint(f"Frames: {count}", force=False)
# end

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
