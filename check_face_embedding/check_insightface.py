import cv2
from insightface.app import FaceAnalysis


def main():
    fa = FaceAnalysis(
        name="antelopev2", root=".insightface",
        providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
    )
    image_path = r"D:\Projects.github\python_projects\check_face_embedding\.maurizio_dataset\2\2_0_DONE\face\20260506_093637_crop_no_margin.jpg"
    img = cv2.imread(image_path)
    faces = fa.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    emb = faces[0].embedding
    pass


if __name__ == "__main__":
    main()
