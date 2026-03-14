import math
from typing import Union, Tuple

import numpy as np
import cv2
import cv2 as cv
import mediapipe as mp
from mediapipe import ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from stdlib.tprint import tprint

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 10

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# ---------------------------------------------------------------------------

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

ROW_SIZE = 10  # pixels
TEXT_COLOR = (255, 0, 0)  # red

# ---------------------------------------------------------------------------

hand_landmark_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_landmark_options = vision.HandLandmarkerOptions(base_options=hand_landmark_base_options, num_hands=2)
hand_landmark_detector = vision.HandLandmarker.create_from_options(hand_landmark_options)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image
# end

# ---------------------------------------------------------------------------

face_detection_base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
face_detection_options = vision.FaceDetectorOptions(base_options=face_detection_base_options)
face_detection_detector = vision.FaceDetector.create_from_options(face_detection_options)

def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_face_detection(
        rgb_image,
        detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      rgb_image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = rgb_image.copy()
    height, width, _ = rgb_image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


# ---------------------------------------------------------------------------

def main():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    vc.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    vc.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
        frame = cv.flip(frame, 1)
    else:
        rval = False
        frame = None

    count = 0
    while rval:
        rval, frame = vc.read()
        if not rval:
            tprint("No frames available")
            break
        frame: np.ndarray = cv.flip(frame, 1)

        image = mp.Image(ImageFormat.SRGB, frame)

        # detection_result = handle_landmark_detector.detect(image)
        # ed_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        detection_result = face_detection_detector.detect(image)
        ed_image = draw_face_detection(image.numpy_view(), detection_result)


        cv2.imshow("preview", ed_image)

        count += 1
        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break
    # end

    vc.release()
    cv2.destroyWindow("preview")


# end


if __name__ == "__main__":
    main()
