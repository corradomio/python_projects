from path import Path as path

# {
#     "scene_id": 0,
#     "drop_base": 1.8594848062328655,
#     "drop_height": 0.24962012421955482,
#     "contact_angle": -3344.4765858059245,
#     "drop_angle": 59.94292450496693,
#     "camera_shift_z": 0.8435633093643125,
#     "scene_shift_z": 0.9610647597278833,
#     "drop_radius": 1.8562829128171445,
#     "drop_shift_x": 0.08652865336743853,
#     "drop_shift_y": 0.246698559941313,
#     "drop_shift_z": -1.6066627885975895,
#     "table_rot_z": 0.8373057688363157,
#     "table_shift_z": 0,
#     "dispenser_side": 0,
#     "dispenser_shift_x": 0,
#     "dispenser_shift_y": 0,
#     "dispenser_shift_z": 0,
#     "drop_bbox": [
#         [
#             63,
#             27
#         ],
#         [
#             68,
#             64
#         ]
#     ],
#     "image_data": [
#         96,
#         96
#     ]
# }


IMAGE_ROOT=path(r"E:\Datasets\WaterDrop\cropped")
ANNOTATIONS_ROOT=path(r"E:\Datasets\WaterDrop\annotations")


def main():
    for dir in IMAGE_ROOT.dirs():
        for jdata in dir.files("*.json"):
            print(jdata)



if __name__ == "__main__":
    main()