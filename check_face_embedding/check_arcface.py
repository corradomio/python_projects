import cv2
from arcface import ArcFace

face_rec = ArcFace.ArcFace()

# cv2.imread("test.png")

image_path = r"D:\Projects.github\python_projects\check_face_embedding\.maurizio_dataset\2\2_0_DONE\face\20260506_093637_crop_no_margin.jpg"
emb1 = face_rec.calc_emb(image_path)

print(emb1)

