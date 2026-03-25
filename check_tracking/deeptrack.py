import cv2 as cv
# from sixdrepnet360 import SixDRepNet360
import face_detection
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics.engine.results import Results

from sixdrepnetx import SixDRepNetMulti
from yolox import YOLOTracking, YOLOPose
from face_detectionx import FaceDetection

# from stdlib.tprint import tprint

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

SIXDREPNET_MULTI = SixDRepNetMulti(dict_path="./models/6DRepNet_300W_LP_AFLW2000.pth")

# SIXDREPNET360 = SixDRepNet360(dict_path="./models/6DRepNet360_300W_LP.pth")

# ['DSFDDetector', 'RetinaNetResNet50', 'RetinaNetMobileNetV1']
FACE_DETECTION = FaceDetection("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, YOLO26l-pose, YOLO26x-pose
POSE_ESTIMATION = YOLOPose("./models/yolo26n-pose.pt")

HUMAN_TRACKING = YOLOTracking("./models/yolo26n.pt")  # Load an official Detect model

TRACKING_ARGS = {
    'batch': 1, 'conf': 0.1, 'mode': 'track', 'show': False, 'tracker': 'botsort.yaml', 'verbose': False,
    'stream': False, 'rect': True, 'classes': [0]
}

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

# def human_tracking_draw(frame, results: list[Results]):
#     img_track = frame.copy()
#     for r in results:
#         img_track = r.plot(img=img_track)
#     frame[:,:,:] = img_track


def mtcnn_net_draw(frame, detection, show_landmarks=True):
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


def face_detection_draw(frame, boxes):
    def _box(box):
        tx, ty, bx, by, prob = box
        top_left_corner = (int(tx), int(ty))
        bottom_right_corner = (int(bx), int(by))
        cv.rectangle(frame, top_left_corner, bottom_right_corner, LINE_COLOR, LINE_THICKNESS)

    for box in boxes:
        _box(box)
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


IMAGE_RGB = "RGB"
N_FRAMES = 1


def stream(vc: cv.VideoCapture):
    count = -1
    while True:
        count += 1
        rval, frame = vc.read()
        if not rval or frame is None:
            continue

        if count % N_FRAMES != 0:
            cv.imshow("preview", frame)
            continue

        h, w, c = frame.shape

        # frame = cv.resize(frame, (w // 2, h // 2))
        ftemp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        annotated = frame.copy()
        image = Image.fromarray(ftemp, IMAGE_RGB)

        # ---------------------------------------------------------------
        # tracking

        results: list[Results] = HUMAN_TRACKING.predict(frame, **TRACKING_ARGS)
        HUMAN_TRACKING.plot(annotated, results)
        track_boxes: dict[int, list[float]] = HUMAN_TRACKING.track_boxes(results)

        # ---------------------------------------------------------------
        # face detector + landmarks

        # boxes, _ = MTCNN_NET.detect(image)
        # boxes = MTCNN_NET.detect(image, landmarks=True)
        # mtcnn_net_draw(frame, boxes)

        # ---------------------------------------------------------------
        # face detector

        detections = FACE_DETECTION.detect(frame)
        FACE_DETECTION.plot(annotated, detections)
        face_detection_draw(annotated, detections)

        # ---------------------------------------------------------------
        # face direction

        pitch, yaw, roll, tdx, tdy = SIXDREPNET_MULTI.predict(frame, detections)
        SIXDREPNET_MULTI.plot(annotated, pitch, yaw, roll, tdx, tdy)

        # ---------------------------------------------------------------

        cv.imshow("preview", annotated)

        # ---------------------------------------------------------------

        key = cv.waitKey(1)
        while key == 32:
            key = cv.waitKey(500)
            key = 32 if key != 32 else 0
        if key == 27:  # exit on ESC
            break
    # end
# end