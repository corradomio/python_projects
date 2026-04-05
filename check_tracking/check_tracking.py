import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics.engine.results import Results

from face_detectionx import FaceDetection
from sixdrepnetx import SixDRepNetMulti
from yolox import YOLOTracking, YOLOPose
from track_saver import TrackSaver
from stdlib.tprint import tprint

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
    'stream': True, 'rect': True, 'classes': [0]
}

TRACK_SAVER = TrackSaver("./faces")


IMAGE_RGB = "RGB"
N_FRAMES = 1
SKIP_FRAMES = 24*120


# def convert_scale(img, alpha, beta):
#     """Add bias and gain to an image with saturation arithmetics. Unlike
#     cv2.convertScaleAbs, it does not take an absolute value, which would lead to
#     nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
#     becomes 78 with OpenCV, when in fact it should become 0).
#     """
#
#     new_img = img * alpha + beta
#     new_img[new_img < 0] = 0
#     new_img[new_img > 255] = 255
#     return new_img.astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0,0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

#
# image = cv2.imread('1.jpg')
# auto_result, alpha, beta = automatic_brightness_and_contrast(image)
# print('alpha', alpha)
# print('beta', beta)
# cv2.imshow('auto_result', auto_result)
# cv2.waitKey()
#


def main():
    # vc = cv2.VideoCapture(0)
    vc = cv2.VideoCapture(r"E:\Movies\Sokurov - 2002 - Arca Russa.mkv")

    tprint("Start stream ...")

    count = -1
    while True:
        count += 1
        rval, frame = vc.read()
        if not rval or frame is None:
            continue

        # tprint(f"preview: {count}", force=False)

        if count < SKIP_FRAMES:
            continue

        h, w, c = frame.shape
        # frame = cv2.resize(frame, (w // 4, h // 4))
        frame = cv2.resize(frame, (w // 2, h // 2))

        # frame, _, _ = automatic_brightness_and_contrast(frame)

        annotated = frame.copy()

        if count % N_FRAMES != 0:
            cv2.imshow("preview", frame)
            continue

        # ftemp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(ftemp, IMAGE_RGB)

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

        euler_rad, td = SIXDREPNET_MULTI.predict_angles(frame, face_detections)
        SIXDREPNET_MULTI.plot_angles(annotated, euler_rad, td)

        # ---------------------------------------------------------------

        # TRACK_SAVER.save(frame, track_boxes, face_detections, euler_rad)

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

