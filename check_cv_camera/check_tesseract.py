import pytesseract
from pytesseract import Output
from PIL import Image
import cv2

# img_path1 = 'Screenshot 2026-04-24 164353.png'
# img_path1 = 'image-with-timestamp.jpg'
img_path1 = '20260218_121637_whole.jpg'
img_path1 = '20260326_120717_whole.jpg'

# text = pytesseract.image_to_string(img_path1, lang='eng')
# print(text)


import pytesseract_api
from pytesseract_api.api import set_variable

img = cv2.imread(img_path1)
text = pytesseract_api.image_to_string(img, lang='eng')
print(text)


