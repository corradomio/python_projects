import sys
import traceback

from human.arcface import ArcFace, ARCFACE_MODEL_NAMES

# cv2.imread("test.png")

image_path = r"D:\Projects.github\python_projects\check_face_embedding\.maurizio\2\2_0_DONE\face\20260506_093637_crop_no_margin.jpg"

for model_name in ARCFACE_MODEL_NAMES:
    print("---", model_name, "---")
    try:
        emb1 = ArcFace.represent(image_path, model_name)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
    pass


# print(emb1)

