from stdlib.tprint import tprint
import cv2 as cv
import cvx

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 10

cv.namedWindow("preview")
vc = cv.VideoCapture(0)

vc.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
vc.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
vc.set(cv.CAP_PROP_FPS, CAMERA_FPS)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv.flip(frame, 1)
else:
    rval = False
    frame = None

cvx.print_props(vc)

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
    cv.imshow("preview", frame)

    count += 1
    key = cv.waitKey(1)
    if key == 27: # exit on ESC
        break

    tprint(f"Frames: {count}, frame_id: {vc.get(cv.CAP_PROP_POS_FRAMES)}", force=False, )
# end

vc.release()
vc.destroyWindow("preview")
