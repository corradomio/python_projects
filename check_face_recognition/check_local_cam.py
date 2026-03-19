import cv2 as cv
import torch
from pprint import pprint
from PIL import Image
from facenet_pytorch import MTCNN
from sixdrepnet import SixDRepNet
from ultralytics import YOLO
import face_detection
import cvx
from stdlib.tprint import tprint

tprint(f"torch: {torch.__version__}")
print(face_detection.available_detectors)

fontFace = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
radius = 2
thickness = 1
lineType = cv.LINE_AA

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 10

cv.namedWindow("preview")

MTCNN_NET = MTCNN(keep_all=True)
# RESNET = InceptionResnetV1(pretrained='casia-webface').eval()
SIXDREPNET = SixDRepNet(dict_path="./6DRepNet_300W_LP_AFLW2000.pth")

# ['DSFDDetector', 'RetinaNetResNet50', 'RetinaNetMobileNetV1']
FACE_DETECTOR = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, YOLO26l-pose, YOLO26x-pose
POSE_ESTIMATION = YOLO("yolo26n-pose.pt")

TRACKING_MODEL = YOLO()

# RGB -> BGR

def mn_draw_boxes(frame, detection, show_landmarks=False):
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

    tprint(detection)

    if isinstance(detection, tuple):
        boxes, accuracies, landmarks = detection
        if boxes is None:
            tprint("NO BOXES")
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


def main():
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

    count = 0
    while rval:
        rval, frame = vc.read()
        if not rval or frame is None:
            tprint("No frames available")
            break

        count += 1

        # "BGR"  NOT SUPPORTED
        # image = Image.fromarray(frame, "BGR")

        # 1) frame in BGR (h,w,c) !!!  is 'c' in BGR
        # 2) convert BGR -> RGB
        # 3) image in RGB
        # pillow modes:  https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        ftemp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(ftemp, "RGB")
        # ftemp = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # image = Image.fromarray(ftemp, "L")

        # MTCNN_NET requires a COLOR image
        # boxes, _ = MTCNN_NET.detect(image)
        # boxes = MTCNN_NET.detect(image, landmarks=True)
        # mn_draw_boxes(frame, boxes, show_landmarks=True)
        # if boxes is not None:
        #     # (n_faces, n_channels, width?, height?)
        #     faces = MTCNN_NET(image)

        # pitch, yaw, roll = SIXDREPNET.predict(frame)
        # pitch, yaw, roll = SIXDREPNET.predict(image)
        # print(f"pitch={pitch}, yaw={yaw}, roll={roll}")

        # detections = FACE_DETECTOR.detect(frame)
        # if count%1 == 0:
        #     fd_draw_boxes(frame, detections)

        poses = POSE_ESTIMATION(frame, verbose=False)
        yolo_draw_poses(frame, poses)
        # if poses is not None and count%100 == 0:
        #     pprint(poses)


        # {do something with the frame here}
        if count == 100:
            image.save(f"image-{count}.png")

        # frame = cv.flip(frame, 1)
        cv.imshow("preview", frame)

        key = cv.waitKey(1)
        if key == 27: # exit on ESC
            break

        tprint(f"Frames: {count}, frame_id: {vc.get(cv.CAP_PROP_POS_FRAMES)}", force=False, )
    # end

    vc.release()
    cv.destroyWindow("preview")
# end


if __name__ == "__main__":
    main()
