import cv2
import stdlib.xmlx as xmlx
from stdlib.dictx import dict_get

VOC_ROOT = "E:/Datasets/VOCdevkit"
VOC_2007 = f"{VOC_ROOT}/VOC2007"

def main():
    # img_path = f"{VOC_2007}/JPEGImages/000001.jpg"
    ann_path = f"{VOC_2007}/Annotations/000001.xml"

    # img = cv2.imread(img_path)
    ann = xmlx.load(ann_path)

    filename = dict_get(ann, ["annotation", "filename"])
    img_path = f"{VOC_2007}/JPEGImages/{filename}"
    img = cv2.imread(img_path)

    objects = dict_get(ann, ["annotation", "object"])
    if isinstance(objects, dict):
        objects = [objects]

    for obj in objects:
        xmin = int(dict_get(obj, ["bndbox", "xmin"]))
        ymin = int(dict_get(obj, ["bndbox", "ymin"]))
        xmax = int(dict_get(obj, ["bndbox", "xmax"]))
        ymax = int(dict_get(obj, ["bndbox", "ymax"]))

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    cv2.imwrite("r000001.jpg", img)

    pass


if __name__ == "__main__":
    main()



