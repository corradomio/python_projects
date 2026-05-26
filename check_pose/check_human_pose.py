from pathlib import Path
import cv2
import torch
import torchvision.transforms.v2
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_keypoints
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# torchvision ???
# ---------------------------------------------------------------------------
# ARE the 17 keypoints the same than YOLO???
#  0. Nose
#  1. Left Eye
#  2. Right Eye
#  3. Left Ear
#  4. Right Ear
#  5. Left Shoulder
#  6. Right Shoulder
#  7. Left Elbow
#  8. Right Elbow
#  9. Left Wrist
# 10. Right Wrist
# 11. Left Hip
# 12. Right Hip
# 13. Left Knee
# 14. Right Knee
# 15. Left Ankle
# 16. Right Ankle
#

TVISION_POSE_MODEL = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
TVISION_POSE_MODEL.eval().to("cuda")

transform = transforms.Compose([
    transforms.v2.ToImage(),
    transforms.v2.ToDtype(torch.float32, scale=True)
])

ToImage = transforms.v2.ToImage()
ToPILImage = transforms.ToPILImage()

# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
#  0. Nose
#  1. Left Eye
#  2. Right Eye
#  3. Left Ear
#  4. Right Ear
#  5. Left Shoulder
#  6. Right Shoulder
#  7. Left Elbow
#  8. Right Elbow
#  9. Left Wrist
# 10. Right Wrist
# 11. Left Hip
# 12. Right Hip
# 13. Left Knee
# 14. Right Knee
# 15. Left Ankle
# 16. Right Ankle
#

# models
# https://platform.ultralytics.com/ultralytics/yolo26
# YOLO26n|s|m|l|x-pose
#
# https://platform.ultralytics.com/ultralytics/yolo11
# YOLO11n|s|m|l|x-pose

YOLO_POSE_MODEL = YOLO("yolo11n-pose.pt")
YOLO_POSE_MODEL.eval().to("cuda")



# ---------------------------------------------------------------------------
# OpenPose
# ---------------------------------------------------------------------------
# https://viso.ai/deep-learning/openpose/
# https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_00_index.html
# https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_03_python_api.html

# to download from
# https://github.com/CMU-Perceptual-Computing-Lab/openpose
#   1.7.0


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

# Detect humans in the image
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map="auto")




# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_glob_walk():
    for img_path in ROOT_PATH.glob("**/*.jpg"):
        print(img_path.name)

    for cdir, dirs, files in ROOT_PATH.walk():
        print(cdir)
        print("...", dirs)
        print("...", files)
    pass


def load_image(image_path: Path):
    image = Image.open(image_path)
    itens = transform(image)
    itens.unsqueeze_(0)
    return itens, ToImage(image)


ROOT_PATH = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-20\20260422_000001")

# ---------------------------------------------------------------------------
# check pose
# ---------------------------------------------------------------------------

def check_torchvision_pose():
    # non si capisce 'na mazza'!
    # perche' su un'immagine contenete SOLO una persona trov 5 keypoints?
    # quali sono i 17 keypoints che trova???

    print("Start scan ...")
    for img_path in ROOT_PATH.glob("**/*.jpg"):
        print(img_path.name)
        img, image = load_image(img_path)
        # img: tensor[N, c, ]
        poses = TVISION_POSE_MODEL(img)
        # poses = [
        #   {
        #       boxes: tensor[(5,4), float32]
        #           N, (x1,y1,x2,y2)
        #       labels: tensor[5, int64]
        #           N, predicted category
        #       scores: tensor[5, float32]
        #           N, confidence
        #       keypoints: tensor[(5, 17, 3), float32]
        #           N, K, (x,y,v)
        #           K: n of keypoints ???
        #           v: visibility
        #       keypoint_scores: tensor[(5, 17), float32]
        #   }
        # ]
        drawn_image = draw_keypoints(
            image,
            keypoints=poses[0]["keypoints"],
            colors="red",
            radius=3
        )
        output_image = ToPILImage(drawn_image)
        output_image.show()
        pass
    pass


def draw_yolo_keypoints(image, keypoints_xy, keypoints_conf, thickness=2):
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image

    # COCO 17-keypoint skeleton (edges between keypoints)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for person_idx, (kpts, confs) in enumerate(zip(keypoints_xy, keypoints_conf)):
        kpts = kpts.cpu().numpy()  # Shape: (17, 2) [x, y]
        confs = confs.cpu().numpy()  # Shape: (17,) [confidence]

        # Draw keypoints
        for i, (x, y) in enumerate(kpts):
            if confs[i] > 0.5:  # Draw keypoints with sufficient confidence
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Draw skeleton lines
        for (start, end) in skeleton:
            if confs[start] > 0.5 and confs[end] > 0.5:
                start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, start_pt, end_pt, (255, 0, 0), thickness)

    return image

def check_yolo_pose():

    print("Start scan ...")
    for img_path in ROOT_PATH.glob("**/*.jpg"):
        if "whole" in img_path.name: continue
        print(img_path.name)
        image = cv2.imread(str(img_path))
        # img, image = load_image(img_path)
        # img: tensor[N, c, ]
        pose_results = YOLO_POSE_MODEL(
            image,
            conf=0.25,
            iou=0.45,
            classes=[0],
            device='cuda:0',
            half=True,
            verbose=False
        )
        # poses = [
        #   Results:
        #       boxes: Boxes[N, 6]
        #       keypoints: Keypoints[(N, 17, 3)]
        #       masks: -
        #       names: {<index>: <class>}
        #       obb: -
        #       orig_img: ndarray[h,w,c]
        #       orig_shape: (h,w)
        #       path: str
        #       probs: -
        #       save_dir: str
        #       speed: {
        #           "preprocess": float,
        #           "inference": float
        #           "postprocess": float
        #       }
        # ]

        # Boxes:
        #   conf: tensor[1, float]
        #   data: tensor[(1, 6), float]
        #   shape: Size
        #   xywh        in pixels
        #   xywhn       in range [0,1]
        #   xyxy        in pixels
        #   xyxyn       in range [0,1]
        #
        # Keypoints:
        #   conf
        #   data
        #   shape
        #   xy
        #   xyn
        #   .

        draw_yolo_keypoints(
            image,
            pose_results[0].keypoints.xy,
            pose_results[0].keypoints.conf
        )

        cv2.imshow("yolo", image)
        if cv2.waitKey(100000) in [ord('.'), ord('q'), 27]:
            break


        pass
    pass


def main():
    # check_torchvision_pose()
    check_yolo_pose()
    pass


if __name__ == "__main__":
    main()
