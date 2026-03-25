from typing import Iterator

import cv2 as cv
from ultralytics import YOLO
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from sixdrepnet import SixDRepNet
from ultralytics.engine.results import Results
from ultralytics.trackers import register_tracker

from sixdrepnet360 import SixDRepNet360
import face_detection
from stdlib.tprint import tprint

FONT_FACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
CIRCLE_RADIUS = 2
FONT_THICKNESS = 1
LINE_TYPE = cv.LINE_AA

LINE_COLOR = (255, 0, 255)
LINE_THICKNESS = 2

# Load an official or custom model

MTCNN_NET = MTCNN(keep_all=True)

SIXDREPNET = SixDRepNet(dict_path="./6DRepNet_300W_LP_AFLW2000.pth")

SIXDREPNET360 = SixDRepNet360(dict_path="./6DRepNet360_300W_LP.pth")

# ['DSFDDetector', 'RetinaNetResNet50', 'RetinaNetMobileNetV1']
FACE_DETECTION = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, YOLO26l-pose, YOLO26x-pose
POSE_ESTIMATION = YOLO("yolo26n-pose.pt")

HUMAN_TRACKING = YOLO("yolo26n.pt")  # Load an official Detect model
register_tracker(HUMAN_TRACKING, False)

# TRACKING_ARGS =  {
#         'batch': 1,
#         'conf': 0.1,
#         'data': '/home/lq/codes/ultralytics/ultralytics/cfg/datasets/coco.yaml',
#         'imgsz': 640,
#         'mode': 'track',
#         'model': 'yolo26n.pt',
#         'rect': True,
#         'save': False,
#         'show': True,
#         'single_cls': False,
#         'task': 'detect',
#         'tracker': 'botsort.yaml',
#         'verbose': False
#     }
# {"batch": 1, "conf": 0.25, "mode": "predict", "save": False, "rect": True}
# {"batch": 1, "conf": 0.1,  "mode": "track",   "show": True, "tracker": "botsort.yaml", "verbose": False}
# {'batch': 1, 'conf': 0.1, 'mode': 'track', 'show': True, 'tracker': 'botsort.yaml', 'verbose': False}
# {"batch": 1, "conf": 0.25, "mode": "predict", "save": False, "rect": True}
# {'batch': 1, 'conf': 0.1,  'mode': 'track',   'show': True, 'tracker': 'botsort.yaml', 'verbose': False}

TRACKING_ARGS = {
    'batch': 1, 'conf': 0.1, 'mode': 'track', 'show': False, 'tracker': 'botsort.yaml', 'verbose': False,
    'stream': False, 'rect': True, 'classes': [0]
}

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def mn_draw_boxes(frame, detection, show_landmarks=True):
    def _box(box):
        tx, ty, bx, by = box
        top_left_corner = (int(tx), int(ty))
        bottom_right_corner = (int(bx), int(by))
        cv.rectangle(frame, top_left_corner, bottom_right_corner, LINE_COLOR, LINE_THICKNESS)

    def _keypoint(keypoints):
        for i, p in enumerate(keypoints):
            x, y = p
            c = (int(x), int(y))
            cv.circle(frame, c, CIRCLE_RADIUS, LINE_COLOR, LINE_THICKNESS)
            cv.putText(frame, str(i), c, FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, LINE_TYPE)
        # end
    # end

    if detection is None:
        return

    if isinstance(detection, tuple):
        boxes, accuracies, landmarks = detection
        if boxes is None:
            # tprint("NO BOXES")
            return
        for box in boxes:
            _box(box)
        if show_landmarks:
            for keypoint in landmarks:
                _keypoint(keypoint)
    else:
        for box in detection:
            _box(box)
# end


def fd_draw_boxes(frame, detection):
    def _box(box):
        tx, ty, bx, by, prob = box
        top_left_corner = (int(tx), int(ty))
        bottom_right_corner = (int(bx), int(by))
        cv.rectangle(frame, top_left_corner, bottom_right_corner, LINE_COLOR, LINE_THICKNESS)

    for box in detection:
        _box(box)
# end


def yolo_draw_poses(frame, poses):
    def _box(box):
        tx, ty, bx, by = box
        top_left_corner = (int(tx), int(ty))
        bottom_right_corner = (int(bx), int(by))
        color = (255, 0, 255)
        thickness = 2
        cv.rectangle(frame, top_left_corner, bottom_right_corner, color, thickness)

    def _keypoint(keypoints):
        for  k in range(keypoints.xy.shape[0]):
            for i in range(17):
                x, y = keypoints.xy[k,i]
                c = (int(x), int(y))
                cv.circle(frame, c, CIRCLE_RADIUS, FONT_COLOR, FONT_THICKNESS)
                cv.putText(frame, str(i), c, FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, LINE_TYPE)

    for pose in poses:
        for boxes in pose.boxes:
            n_boxes = boxes.shape[0]
            for b in range(n_boxes):
                _box(boxes[b].xyxy[0])

        _keypoint(pose.keypoints)
# end


def yolo_draw_rects(frame, results: list[Results]):
    # def _box(box):
    #     tx, ty, bx, by = box.to('cpu').numpy()
    #     top_left_corner = (int(tx), int(ty))
    #     bottom_right_corner = (int(bx), int(by))
    #     color = (255, 0, 255)
    #     thickness = 2
    #     cv.rectangle(frame, top_left_corner, bottom_right_corner, color, thickness)

    for r in results:
        ret = r.plot()
        # for b in r.boxes:
        #     for i in range(b.shape[0]):
        #         bi = b[i]
        #
        #         for j in range(bi.xyxy.shape[0]):
        #             bi.
        #             _box(b[i].xyxy[j])
        #     pass
        # return ret
        frame[:,:,:] = ret
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

N_FRAMES = 1

def main():
    # vc = cv.VideoCapture(r"E:\Movies\FILM - Polar - 2019.iTALiAN.WEBRiP.XviD-PRiME.avi")
    vc = cv.VideoCapture(0)

    count = -1
    rval = True
    while rval:
        count += 1
        rval, frame = vc.read()

        if count % N_FRAMES != 0:
            cv.imshow("preview", frame)
            continue

        h, w, c = frame.shape
        frame = cv.resize(frame, (w//2, h//2))
        # ftemp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # image = Image.fromarray(ftemp, "RGB")

        # --- face detector

        # boxes, _ = MTCNN_NET.detect(image)
        # boxes = MTCNN_NET.detect(image, landmarks=True)
        # mn_draw_boxes(frame, boxes)

        # -- face detector

        # detections = FACE_DETECTOR.detect(frame)
        # fd_draw_boxes(frame, detections)

        # ---

        # poses = POSE_ESTIMATION(frame, verbose=False)
        # yolo_draw_poses(frame, poses)

        # ---

        pitch, yaw, roll = SIXDREPNET.predict(frame)
        SIXDREPNET.draw_axis(frame, pitch, yaw, roll)

        # ---

        # pitch_yaw_roll = SIXDREPNET360.predict(frame)

        # ---

        # results: list[Results] = HUMAN_TRACKING.predict(frame, **TRACKING_ARGS)
        # yolo_draw_rects(frame, results)

        # for track in tracks:
        #     # (240, 320, 3)
        #     frame = track.orig_img
        #     # print(track.boxes.data)  # print detection bounding boxes

        # ---

        cv.imshow("preview", frame)

        key = cv.waitKey(1)
        while key == 32:
            key = cv.waitKey(500)
            key = 32 if key != 32 else 0
        if key == 27:  # exit on ESC
            break
    # end
# end


if __name__ == "__main__":
    main()
