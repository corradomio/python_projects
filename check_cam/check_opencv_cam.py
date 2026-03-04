from stdlib.tprint import tprint
import cv2
import cv2 as cv
import cvx

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 10

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

vc.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
vc.set(cv2.CAP_PROP_FPS, CAMERA_FPS)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv.flip(frame, 1)
else:
    rval = False
    frame = None

for prop, pname in cvx.VIDEO_CAPTURE_PROPERTIES.items():
    value = vc.get(prop)
    if value > 0:
        print(f"{pname}: {value}")


f = None
count = 0
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
    frame = cv.flip(frame, 1)
    cv2.imshow("preview", frame)

    count += 1
    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break
    tprint(f"Frames: {count}", force=False)
# end

vc.release()
cv2.destroyWindow("preview")
