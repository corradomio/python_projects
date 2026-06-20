__all__ = [
    "classify_image"
]

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

conf_dress_thr = 0.85
conf_glove_thr = 0.25
conf_cleaner_security_thr = 0.92

IOU_THRESHOLD = 0.5
conf_thr = 0.2


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# classes
# 0 = {str} 'blue_dress'
# 1 = {str} 'blue_glove'
# 2 = {str} 'white_dress'
# 3 = {str} 'black_glove'
# 4 = {str} 'cleaner'
# 5 = {str} 'security'
# 6 = {str} 'maintenance'
# 7 = {str} 'white_glove'
#
DRESS_GLOVE_MODEL = YOLO("yolo_models/lab_dress_glove.pt").to(TORCH_DEVICE).eval()



def classify_image(image_file: str) -> list[np.ndarray]:
    """
    Classify an image. It return 3 classifications

        1) [person, cleaner_or_security]    type of person
        2) [not_well_dress, well_dress]     if the person is wll dressed
        3) [not_glove_well, glove_well]     if the person has the gloves

    :param image_file: path of the image file
    :return: list of classifications
    """
    assert isinstance(image_file, str), "Parameter 'image_file' must be a str"

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = img.shape

    # just to support multiple images in a single call
    images_to_analyze = [img]

    dress_glove_results: list[Results] = DRESS_GLOVE_MODEL(
        images_to_analyze,
        batch=len(images_to_analyze),
        conf=conf_thr,
        iou=IOU_THRESHOLD,
        verbose=False,
        device=TORCH_DEVICE
    )

    result = dress_glove_results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxys = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    n_classes = len(boxes)

    # classes:
    #   dress:  1,3,7   > conf_glove_thr
    #   glove   0, 2    > conf_dress_thr
    #   person: 4, 5, 6 > conf_cleaner_security_thr

    person_type = [0., 1.]
    dress_type  = [1., 0.]
    glove_type  = [1., 0.]

    for i in range(n_classes):
        cls = classes[i]
        cnf = confs[i]
        if cls in [1,3,7]:
            dress_type = [1-cnf, cnf]
        elif cls in [0,2]:
            glove_type = [1-cnf, cnf]
        else:
            person_type = [1-cnf, cnf]
    # end

    # converted in:
    #   [person, cleaner_or_security]
    #   [not_well_dress, well_dress]
    #   [not_glove_well, glove_well]

    return [
        np.array(person_type),
        np.array(dress_type),
        np.array(glove_type)
    ]

