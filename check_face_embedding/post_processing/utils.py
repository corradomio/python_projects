from pathlib import Path
from typing import Any
import os
import re
import numpy as np
from collections import Counter
from stdlib.is_instance import is_instance

IMAGE_ARRAY = np.ndarray
NP_ARRAY = np.ndarray
EMBEDDING = np.ndarray # [512]
IMAGES_EMBEDDING = list[EMBEDDING]

RECORD = dict[str, Any]
CAM_TRACK_NAME = str            # <cam_id>_<track-id>_DONE
DATE_CAM_TRACK_NAME = str       # <YYYYMMDD>_<cam_id>_<track-id>_DONE
LOCAL_FOLDER = str              #
REMOTE_FOLDER = str
FILE_PATH = str
REMOTE_FILE_PATH = str
PERSON_NAME = str
URL = str

# Sub directories
#
#   face_recognition
#   random_crop
#
#   cleaner_or_security
#   not_dress_well
#   not_glove_well
#   unauthorised_access
#   unauthorised_operation_A
#   unauthorised_machine_touching_B
#

def normalize_path(path: str, end_slash=None) -> str:
    """
    Replace '\\' with '/', and '//' with '/'
    Add the end slash, if required

    :param path: path to normalize
    :param end_slash: if path must terminate with '/'
    :return:
    """
    path = path.replace("\\","/")
    while "//" in path:
        path = path.replace("//", "/")
    if end_slash is None:
        pass
    elif end_slash:
        if not path.endswith("/"):
            path += "/"
    else:
        if path.endswith("/"):
            path = path[:-1]
    return path


def parent_of(path: str) -> str:
    """
    Parent of the (normalized) path
    :param path: path
    :return: parent path
    """
    pos = path.rfind("/")
    return path[:pos] if pos > 0 else ""


def name_of(path: str, ext:bool=True) -> str:
    """
    Name of the last component of the (normalized) path.
    Note: path MUST BE NORMALIZED ('\\'->'/')
    :param path: path
    :param ext: if to exclude the extension
    :return: name of the path
    """
    assert "\\" not in path
    # remove last slash
    if path.endswith("/"):
        path = path[:-1]
    pos = path.rfind("/")
    name = path[pos+1:]
    end = -1
    if not ext:
        end = name.rfind(".")
    if end != -1:
        name = name[:end]
    return name


def folder_of(img_files: list[FILE_PATH]) -> str:
    """
    Extract the <folder> component from a path having the structure:

        .../<folder>/face_recognition/<image_file_name>.<ext>  ->  <folder>

    :param img_files:
    :return:
    """
    assert isinstance(img_files, list)
    if len(img_files) == 0:
        return ""
    img_file: str = normalize_path(img_files[0])
    # .../<folder>/face_recognition/<image_file_name>.<ext>
    pos = img_file.rfind("/")
    img_file = img_file[:pos]
    # .../<folder>/face_recognition
    pos = img_file.rfind("/")
    img_file = img_file[:pos]
    # .../<folder>
    pos = img_file.rfind("/")
    folder = img_file[pos+1:]
    return folder


def image_timestamp_of(img_file: str) -> str:
    # .../<image_file_name>.<ext>  ->  <yyyy><mm><dd>_<HH><MM><SS>
    # where <image_file_name> has the following structure
    #
    #   <yyyy><mm><dd>_<HH><MM><SS>_<suffix>.<ext>
    #
    # <suffix>: one of
    #   crop_no_margin
    #   crop
    #   whole
    #
    assert is_instance(img_file, str)
    image_name: str = normalize_path(img_file)
    pos = img_file.rfind("/")
    if pos != -1:
        image_name = image_name[pos+1:]
    pos = image_name.find("_", 10)         # <yyyy><mm><dd>_...
    assert pos == 15                      # <yyyy><mm><dd>_<HH><MM><SS>_...
    image_name = image_name[:pos]
    return image_name


def list_camera_folders(root_path: Path, cam_id: int) -> list[Path]:
    assert is_instance(root_path, Path)
    assert is_instance(cam_id, int)
    sub_folders_this_cam = []
    pattern_done = re.compile(rf'^{cam_id}_\d+_DONE$')
    for subfolder in root_path.iterdir():
        if pattern_done.match(subfolder.name):
            sub_folders_this_cam.append(subfolder)

    assert is_instance(sub_folders_this_cam, list[Path])
    return sub_folders_this_cam


def list_images(folder: Path, ext=".jpg") -> list[Path]:
    assert is_instance(folder, Path)
    if folder.exists():
        return [img  for img in folder.iterdir() if img.name.endswith(ext)]
    else:
        return []



def is_timestamp_folder(name: str):
    """
    check if the name of the folder is YYYYMMDD_hhmmss
    :param name:
    :return:
    """
    if name.endswith("_DONE"):
        return False
    name = name_of(name)
    parts = name.split('_')
    if len(parts) != 2:
        return False
    if len(parts[0]) != 8 or len(parts[1]) != 6:
        return False
    return True
# end


def is_root_folder_empty(root_folder: str) -> bool:
    for dir in os.listdir(root_folder):
        if is_timestamp_folder(dir):
            continue
        return False
    return True


def most_common_person_name(names_list: list[str]):
    if names_list is None or len(names_list) == 0:
        return "NO_FACES_SAVED"
    ignores = [ "face_not_in_DB", "NO_FACES_SAVED"]
    filtered_names = [name for name in names_list if name not in ignores]
    if not filtered_names or len(filtered_names) == 0:
        if "face_not_in_DB" in names_list:
            return "face_not_in_DB"
        else:
            return "NO_FACES_SAVED"
    counts = Counter(filtered_names)
    name, count = counts.most_common(1)[0]
    return name


def chop(x, xmax: int):
    if x < 0:
        x = 0
    elif x > xmax:
        x = xmax
    return int(x)

