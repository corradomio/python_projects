from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests

# url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

initial_image_path = r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_mini\2026-07-21-00-00\0_6608_DONE\face\face_20260721_082700_crop_no_margin.jpg"

image = Image.open(initial_image_path)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')
