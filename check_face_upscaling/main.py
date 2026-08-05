import os

import cv2
import requests
import torch
from PIL import Image
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

# ---------------------------------------------------------------------------

model_urls = {
    'realesr-general-x4v3.pth': "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    'GFPGANv1.4.pth': "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}

os.makedirs('weights', exist_ok=True)

def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

for filename, url in model_urls.items():
    file_path = os.path.join('weights', filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        download_file(url, file_path)
    else:
        print(f"{filename} already exists. Skipping download.")

# ---------------------------------------------------------------------------

print(os.listdir('weights'))

# ---------------------------------------------------------------------------

realesrgan_model_path = 'weights/realesr-general-x4v3.pth'

sr_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
realesrganer = RealESRGANer(scale=4, model_path=realesrgan_model_path, model=sr_model, tile=0, tile_pad=10, pre_pad=0, half=half)

def upscale_image(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    output, _ = realesrganer.enhance(img, outscale=4)
    cv2.imwrite(output_path, output)
    return output

# ---------------------------------------------------------------------------

gfpgan_model_path = 'weights/GFPGANv1.4.pth'

face_enhancer = GFPGANer(model_path=gfpgan_model_path, upscale=10, arch='clean', channel_multiplier=2, bg_upsampler=realesrganer)

# Function to enhance image with GFPGAN
def enhance_faces(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    _, _, img_enhanced = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    cv2.imwrite(output_path, img_enhanced)
    return img_enhanced


# --------------------------------------------------------------------------
# initial_image_path = '/kaggle/input/old-photos/old_photo_01.jpg'
initial_image_path = r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_mini\2026-07-21-00-00\0_6608_DONE\face\face_20260721_082700_crop_no_margin.jpg"

# Load the image with PIL
photo = Image.open(initial_image_path)
W, H = photo.size

photo.resize((W*10, H*10)).show("Image")

# display(photo.resize((800, 800), Image.LANCZOS))

enhanced_faces_path = "enhanced_faces.jpg"

try:
    enhance_faces(initial_image_path, enhanced_faces_path)
    enhanced_image_to_display = Image.open(enhanced_faces_path)
    # display(enhanced_image_to_display)
    enhanced_image_to_display.show("Enhanced")

except Exception as err:
    print(f"An error occurred during face enhancement: {err}")