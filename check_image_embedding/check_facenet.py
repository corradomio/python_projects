import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=128, margin=10)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

image_path1 = r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-16\0_3_DONE\face_recognition\20260218_141814_crop_no_margin.jpg"
# image_path2 = r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-16\0_4_DONE\random_crop\20260218_141821_crop_no_margin.jpg"
image_path2 = r"D:\Projects.ebtic\project.diwang\lab_monitoring\.data_and_result\2026-05-16\0_186_DONE\face_recognition\20260326_150418_crop_no_margin.jpg"


img1 = Image.open(image_path1)
img2 = Image.open(image_path2)

# Get cropped and prewhitened image tensor
img1_cropped = mtcnn(img1, save_path="./face1.png")
img2_cropped = mtcnn(img2, save_path="./face2.png")

# Calculate embedding (unsqueeze to add batch dimension)
img1_embedding = resnet(img1_cropped.unsqueeze(0))
img2_embedding = resnet(img2_cropped.unsqueeze(0))

print(img1_embedding.shape)
print(img2_embedding.shape)

print(torch.dot(img1_embedding[0], img2_embedding[0]).float())

print()

# Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))

