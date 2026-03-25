import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics.engine.results import Results

from face_detectionx import FaceDetection
from sixdrepnetx import SixDRepNetMulti
from yolox import YOLOTracking, YOLOPose
from track_saver import TrackSaver

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
CIRCLE_RADIUS = 2
FONT_THICKNESS = 1
LINE_TYPE = cv2.LINE_AA

LINE_COLOR = (255, 0, 255)
LINE_THICKNESS = 2

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

TRACK_SAVER = TrackSaver("./faces")


IMAGE_RGB = "RGB"
N_FRAMES = 1


def main():
    vc = cv2.VideoCapture(0)

    count = -1
    while True:
        count += 1
        rval, frame = vc.read()
        if not rval or frame is None:
            continue

        if count % N_FRAMES != 0:
            cv2.imshow("preview", frame)
            continue

        h, w, c = frame.shape

        # frame = cv2.resize(frame, (w // 2, h // 2))
        ftemp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        annotated = frame.copy()
        image = Image.fromarray(ftemp, IMAGE_RGB)

        # ---------------------------------------------------------------
        # tracking

        results: list[Results] = HUMAN_TRACKING.predict(frame, **TRACKING_ARGS)
        HUMAN_TRACKING.plot(annotated, results)
        track_boxes: dict[int, np.ndarray] = HUMAN_TRACKING.track_boxes(results)

        # ---------------------------------------------------------------
        # face detector + landmarks

        # boxes, _ = MTCNN_NET.detect(image)
        # boxes = MTCNN_NET.detect(image, landmarks=True)
        # mtcnn_net_draw(frame, boxes)

        # ---------------------------------------------------------------
        # face detector

        face_detections = FACE_DETECTION.detect(frame)
        FACE_DETECTION.plot(annotated, face_detections)

        # ---------------------------------------------------------------
        # face direction

        # pitch, yaw, roll, tdx, tdy = SIXDREPNET_MULTI.predict(frame, face_detections)
        # SIXDREPNET_MULTI.plot(annotated, pitch, yaw, roll, tdx, tdy)
        euler_rad, td = SIXDREPNET_MULTI.predict_angles(frame, face_detections)
        SIXDREPNET_MULTI.plot_angles(annotated, euler_rad, td)

        # ---------------------------------------------------------------

        TRACK_SAVER.save(frame, track_boxes, face_detections, euler_rad)

        # ---------------------------------------------------------------

        cv2.imshow("preview", annotated)

        # ---------------------------------------------------------------

        key = cv2.waitKey(1)
        while key == 32:
            key = cv2.waitKey(500)
            key = 32 if key != 32 else 0
        if key == 27:  # exit on ESC
            break
    # end



if __name__ == "__main__":
    main()

