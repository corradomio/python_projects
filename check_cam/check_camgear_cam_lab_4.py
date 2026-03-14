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
stream0 = CamGear(source=all_rtsp_urls[0], **OPTIONS).start()
stream1 = CamGear(source=all_rtsp_urls[1], **OPTIONS).start()
stream2 = CamGear(source=all_rtsp_urls[2], **OPTIONS).start()
stream3 = CamGear(source=all_rtsp_urls[3], **OPTIONS).start()

# loop over
count = 0
while True:
    # read frames from stream
    frame0 = stream0.read()
    frame1 = stream1.read()
    frame2 = stream2.read()
    frame3 = stream3.read()

    if frame0 is not None:
        # frame0 = cv.flip(frame0, 1)
        cv2.imshow("preview0", frame0)
    if frame1 is not None:
        # frame1 = cv.flip(frame1, 1)
        cv2.imshow("preview1", frame1)
    if frame2 is not None:
        # frame2 = cv.flip(frame2, 1)
        cv2.imshow("preview2", frame2)
    if frame3 is not None:
        # frame3 = cv.flip(frame3, 1)
        cv2.imshow("preview3", frame3)

    count += 1
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # exit on ESC
        break
    tprint(f"Frames: {count}", force=False)
# end

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream3.stop()
stream2.stop()
stream1.stop()
stream0.stop()
