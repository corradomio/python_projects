import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

# If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open(r"D:\Projects.ebtic\project.diwang\lab_monitoring\data_and_result\2026-05-15\20260422_112233\2_5521_DONE\face\20260218_141827_crop_no_margin.jpg")
img = transforms.ToTensor()(img)

# Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path=<optional save path>)
img_cropped = img

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))

print(img_probs)