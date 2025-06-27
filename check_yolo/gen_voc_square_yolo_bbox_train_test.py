import numpy as np
import pandas as pd
from path import Path as path
from random import random
import xml.etree.ElementTree as ET
from stdlib.tprint import tprint

# VOC_ROOT = r"E:/Datasets/VOC2012"
VOC_ROOT = r"D:/Datasets/VOC2012"

LABELS = path(f"{VOC_ROOT}/Annotations")
IMAGES = path(f"{VOC_ROOT}/JPEGImages")
LABELS_YOLO = path(f"{VOC_ROOT}/Annotations_yolo")
LABELS_VOC = path(f"{VOC_ROOT}/Annotations_voc")
LABELS_YOLO_SQUARE = path(f"{VOC_ROOT}/Annotations_yolo_square")


CLASS_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_INDICES = {CLASS_LABELS[i]:i for i in range(len(CLASS_LABELS))}


def xml_to_yolo():
    tprint("xml_to_yolo", force=True)
    data = []
    for f in LABELS.files("*.xml"):
        tprint("..." + f)
        root = ET.parse(f).getroot()

        image_name = root.find("filename").text
        label_name = f.stem + ".txt"

        image_width  = int(root.find("size").find("width").text)
        image_height = int(root.find("size").find("height").text)

        image_width = max(image_width, image_height)
        image_height = image_width

        f_data = []

        for obj in root.findall("object"):
            # <object>
            # 		<name>person</name>
            # 		<bndbox>
            # 			<xmin>277</xmin>
            # 			<ymin>3</ymin>
            # 			<xmax>500</xmax>
            # 			<ymax>375</ymax>
            # 		</bndbox>
            # 	</object>

            name = obj.find("name").text
            class_label = CLASS_INDICES[name]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            xmin, xmax = min(xmin,xmax), max(xmin,xmax)
            ymin, ymax = min(ymin,ymax), max(ymin,ymax)

            #
            # convert coordinates in range [0,1]
            # x,y must be at center!
            #

            x = 0.5*(xmin+xmax)/image_width
            y = 0.5*(ymin+ymax)/image_height
            width  = (xmax - xmin)/image_width
            height = (ymax - ymin)/image_height
            # width = xmax/image_width
            # height = ymax/image_height

            assert 0 <= x <= 1
            assert 0 <= y <= 1
            assert 0 <= width <= 1
            assert 0 <= height <= 1

            #  [x, y, width, height, class_label]
            f_data.append([
                class_label, x, y, width, height
            ])
        # end
        label_path = LABELS_YOLO_SQUARE / label_name
        np.savetxt(label_path, f_data)

        data.append([
            image_name, label_name
        ])
    # end

    data_tt = data[:-5000]
    data_train  = data_tt[:-5000]
    data__test = data[5000:]

    df = pd.DataFrame(data=data, columns=["image_name", "label_name"])
    df_train = pd.DataFrame(data=data_train, columns=["image_name", "label_name"])
    df__test = pd.DataFrame(data=data__test, columns=["image_name", "label_name"])

    df.to_csv(f"{VOC_ROOT}/labels.csv", index=False)
    df_train.to_csv(f"{VOC_ROOT}/train.csv", index=False)
    df__test.to_csv(f"{VOC_ROOT}/test.csv", index=False)
    tprint("done", force=True)
# end


def xml_to_voc():
    tprint("xml_to_voc", force=True)
    data = []
    for f in LABELS.files("*.xml"):
        tprint("..." + f)
        root = ET.parse(f).getroot()

        image_name = root.find("filename").text
        label_name = f.stem + ".txt"

        image_width  = int(root.find("size").find("width").text)
        image_height = int(root.find("size").find("height").text)

        f_data = []

        for obj in root.findall("object"):
            # <object>
            # 		<name>person</name>
            # 		<bndbox>
            # 			<xmin>277</xmin>
            # 			<ymin>3</ymin>
            # 			<xmax>500</xmax>
            # 			<ymax>375</ymax>
            # 		</bndbox>
            # 	</object>

            name = obj.find("name").text
            class_label = CLASS_INDICES[name]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            xmin, xmax = min(xmin,xmax), max(xmin,xmax)
            ymin, ymax = min(ymin,ymax), max(ymin,ymax)

            #
            # convert coordinates in range [0,1]
            # x,y must be at center!
            #

            #  [x, y, width, height, class_label]
            f_data.append([
                class_label, xmin, ymin, xmax, ymax
            ])
        # end
        label_path = LABELS_VOC / label_name
        np.savetxt(label_path, f_data, fmt="%d")

        data.append([
            image_name, label_name
        ])
    # end

    # data_tt = data[:-5000]
    # data_train  = data_tt[:-5000]
    # data__test = data[5000:]

    # df = pd.DataFrame(data=data, columns=["image_name", "label_name"])
    # df_train = pd.DataFrame(data=data_train, columns=["image_name", "label_name"])
    # df__test = pd.DataFrame(data=data__test, columns=["image_name", "label_name"])

    # df.to_csv(f"{VOC_ROOT}/labels.csv", index=False)
    # df_train.to_csv(f"{VOC_ROOT}/train.csv", index=False)
    # df__test.to_csv(f"{VOC_ROOT}/test.csv", index=False)
    tprint("done", force=True)
# end


def main():
    xml_to_yolo()
    xml_to_voc()
    pass




if __name__ == "__main__":
    main()
