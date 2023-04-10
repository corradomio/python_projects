import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="E:\Datasets\ImageSegmentation\sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

img = cv2.imread("frutta-e-verdura.jpg")

masks = mask_generator.generate(img)

pass
