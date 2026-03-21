import cv2 as cv
from ultralytics import YOLO
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from sixdrepnet import SixDRepNet
from sixdrepnet360 import SixDRepNet360
import face_detection
from stdlib.tprint import tprint

fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
radius = 2
thickness = 1
lineType = cv.LINE_AA

# Load an official or custom model

MTCNN_NET = MTCNN(keep_all=True)

SIXDREPNET = SixDRepNet(dict_path="./6DRepNet_300W_LP_AFLW2000.pth")

SIXDREPNET360 = SixDRepNet360(dict_path="./6DRepNet360_300W_LP.pth")

# ['DSFDDetector', 'RetinaNetResNet50', 'RetinaNetMobileNetV1']
FACE_DETECTOR = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, YOLO26l-pose, YOLO26x-pose
POSE_ESTIMATION = YOLO("yolo26n-pose.pt")


def mn_draw_boxes(frame, detection, show_landmarks=True):
    def _box(box):
        tx, ty, bx, by = box
        top_left_corner = (int(tx), int(ty))
        bottom_right_corner = (int(bx), int(by))
        color = (255, 0, 255)
        thickness = 2
        cv.rectangle(frame, top_left_corner, bottom_right_corner, color, thickness)

    def _keypoint(keypoints):
        for i, p in enumerate(keypoints):
            x, y = p
            c = (int(x), int(y))
            cv.circle(frame, c, radius, color, thickness)
            cv.putText(frame, str(i), c, fontFace, fontScale, color, thickness, lineType)
        # end
    # end

    if detection is None:
        return

    # tprint(detection)

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
        color = (255, 0, 255)
        thickness = 2
        cv.rectangle(frame, top_left_corner, bottom_right_corner, color, thickness)

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
                cv.circle(frame, c, radius, color, thickness)
                cv.putText(frame, str(i), c, fontFace, fontScale, color, thickness, lineType)

    for pose in poses:
        for boxes in pose.boxes:
            n_boxes = boxes.shape[0]
            for b in range(n_boxes):
                _box(boxes[b].xyxy[0])

        _keypoint(pose.keypoints)
# end

N_FRAMES = 10


def main():
    vc = cv.VideoCapture(r"E:\Movies\FILM - Polar - 2019.iTALiAN.WEBRiP.XviD-PRiME.avi")

    count = -1
    rval = True
    while rval:
        count += 1
        rval, frame = vc.read()

        if count % N_FRAMES != 0:
            cv.imshow("preview", frame)
            continue

        ftemp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(ftemp, "RGB")

        # ---

        # boxes, _ = MTCNN_NET.detect(image)
        # boxes = MTCNN_NET.detect(image, landmarks=True)
        # mn_draw_boxes(frame, boxes)

        # --

        # detections = FACE_DETECTOR.detect(frame)
        # fd_draw_boxes(frame, detections)

        # ---

        # poses = POSE_ESTIMATION(frame, verbose=False)
        # yolo_draw_poses(frame, poses)

        # ---

        # pitch, yaw, roll = SIXDREPNET.predict(frame)
        # SIXDREPNET.draw_axis(frame, pitch, yaw, roll)

        pitch_yaw_roll = SIXDREPNET360.predict(frame)

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
