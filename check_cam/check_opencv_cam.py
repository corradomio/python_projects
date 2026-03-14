from stdlib.tprint import tprint
import cv2
import cv2 as cv
import cvx

ips = ["10.248.37.111", "10.248.37.108", "10.248.37.12", "10.248.37.100"]
username = "admin"
password = "password1234"
all_rtsp_urls = [f"rtsp://{username}:{password}@{ip}/Streaming/Channels/101?tcp" for ip in ips]

# rtsp://admin:password1234@10.248.37.111/Streaming/Channels/101?tcp

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 10

cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)
vc = cv2.VideoCapture(all_rtsp_urls[0])

vc.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
vc.set(cv2.CAP_PROP_FPS, CAMERA_FPS)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv.flip(frame, 1)
else:
    rval = False
    frame = None

def print_props():
    for prop, pname in cvx.VIDEO_CAPTURE_PROPERTIES.items():
        value = vc.get(prop)
        if value > 0:
            print(f"{pname}: {value}")


f = None
count = 1
while rval:
    rval, frame = vc.read()
    if not rval:
        tprint("No frames available")
        break

    # if vc.grab():
    #     # rval, frame = vc.retrieve()
    #     rval, f = vc.retrieve(frame)
    #     frame = cv.flip(frame, 1)
    #
    #     # rval: true/false
    #     # f: same as frame or a new frame
    # else:
    #     tprint("No frames available")
    #     break

    # {do something with the frame here}
    # frame = cv.flip(frame, 1)
    cv2.imshow("preview", frame)

    count += 1
    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break

    if count % 100 == 0:
        tprint(f"Frames: {count}, frame_id: {vc.get(cv2.CAP_PROP_POS_FRAMES)}", force=False, )
# end

vc.release()
cv2.destroyWindow("preview")
