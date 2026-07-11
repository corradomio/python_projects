import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Union, Optional
from datetime import datetime

import numpy as np

from stdlib.is_instance import is_instance
from stdlib import jsonx
from stdlib.jsonx import JSONConfiguration
from stdlib.sortedx import sort_by_key

# ---------------------------------------------------------------------------
# Type Hints aliases
# ---------------------------------------------------------------------------

IMAGE_ARRAY = np.ndarray
NP_ARRAY = np.ndarray
EMBEDDING = np.ndarray # [512]

CLUSTER_ID = int
RECORD = dict[str, Any]
CAM_TRACK_NAME = str            # <cam_id>_<track-id>_DONE
DATE_CAM_TRACK_NAME = str       # <YYYYMMDD>_<cam_id>_<track-id>_DONE
LOCAL_FOLDER = str              #
REMOTE_FOLDER = str
FILE_PATH = str
REMOTE_FILE_PATH = str
PERSON_NAME = str
URL = str

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

VIOLATIONS = [
    'not_dress_well', 'not_glove_well',
    'unauthorised_access',
    'unauthorised_operation_A',
    'unauthorised_machine_touching_B'
    #, 'cleaner_or_security'
]

NOT_ASSIGNED_NAMES = ["face_not_in_DB", "NO_FACES_SAVED", "KU_FACE_RECOGNITION_UNAVAILABLE"]

METRIC_TYPES = Literal[
            "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
            "hamming", "jaccard", "jensenshannon", "mahalanobis", "matching", "minkowski", "rogerstanimoto",
            "russellrao", "seuclidean", "sokalsneath", "sqeuclidean", "yule"
        ]


# ---------------------------------------------------------------------------
# LabMonitoring
# ---------------------------------------------------------------------------

class LabMonitoring:
    def __init__(self, CONFIG: JSONConfiguration, component: str):
        assert isinstance(CONFIG, JSONConfiguration)
        assert isinstance(component, str)

        self.CONFIG = CONFIG
        self.component = component

        assert component in CONFIG, f"Missing configuration for '{component}'"
        pass


class DataServer(LabMonitoring):
    def __init__(self, CONFIG: JSONConfiguration, component: str):
        super().__init__(CONFIG, component)

    def save_data(
        self,
        tracks_root: Path,
        meta_records_map: dict[CAM_TRACK_NAME, RECORD],
        date_in_id: str,
        to_combine: list[list[CAM_TRACK_NAME]]
    ) -> dict:
        ...

# ---------------------------------------------------------------------------
# TracksCluster
# ---------------------------------------------------------------------------

class TracksClusterBase:
    def __init__(self, cluster_id: int, tracks_root: Path):
        assert is_instance(cluster_id, int)
        assert is_instance(tracks_root, Path)

        self.cluster_id: int = cluster_id
        self.tracks_root: Path = tracks_root

        self.track_names: list[CAM_TRACK_NAME] = []
        self.tracks_embeddings: list[EMBEDDING] = []
        self.cluster_embeddings: list[EMBEDDING] = []

        self._min_timestamp: float = 0.
        self._max_timestamp: float = 0.
    # end
# end


class ClusterTracks(LabMonitoring):
    def __init__(self, CONFIG: JSONConfiguration, component: str):
        super().__init__(CONFIG, component)

    def track_cluster_map(self) -> dict[CAM_TRACK_NAME, CLUSTER_ID]:
        ...

    def cluster_tracks_map(self) -> dict[CLUSTER_ID, TracksClusterBase]:
        ...

    def has_track(self, track_name: str) -> bool:
        ...

    def analyze(self, tracks_root: Path, tracks_meta_map: dict[CAM_TRACK_NAME, RECORD], save: bool = True) -> bool:
        ...
# end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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

def load_track_meta(track_dir: Path) -> dict:
    meta_file = track_dir / "meta.json"
    meta = jsonx.load(meta_file)

    start = datetime.strptime(meta['present_start'], "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(meta['present_end'], "%Y-%m-%d %H:%M:%S")
    meta['present_start_datetime'] = start
    meta['present_end_datetime'] = end
    meta['duration'] = (end - start).total_seconds()
    return meta
# end


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
    assert isinstance(img_file, str)
    image_name: str = normalize_path(img_file)
    pos = img_file.rfind("/")
    if pos != -1:
        image_name = image_name[pos+1:]
    pos = image_name.find("_", 10)         # <yyyy><mm><dd>_...
    assert pos == 15                      # <yyyy><mm><dd>_<HH><MM><SS>_...
    image_name = image_name[:pos]
    return image_name


def list_camera_folders(root_path: Path, cam_id: int) -> list[Path]:
    assert isinstance(root_path, Path)
    assert isinstance(cam_id, int)
    sub_folders_this_cam = []
    pattern_done = re.compile(rf'^{cam_id}_\d+_DONE$')
    for subfolder in root_path.iterdir():
        if pattern_done.match(subfolder.name):
            sub_folders_this_cam.append(subfolder)

    assert is_instance(sub_folders_this_cam, list[Path])
    return sub_folders_this_cam


def list_images(folder: Path, ext=".jpg") -> list[Path]:
    assert isinstance(folder, Path)
    if folder.exists():
        return [img  for img in folder.iterdir() if img.name.endswith(ext)]
    else:
        return []


def is_timestamp_folder(folder: Path):
    """
    check if the name of the folder is <YYYYMMDD_hhmmss>
    :param name:
    :return:
    """
    assert isinstance(folder, Path)
    name = folder.name

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


def is_root_folder_empty(root_folder: Path) -> bool:
    if not root_folder.exists():
        return True
    for sdir in root_folder.iterdir():
        if is_timestamp_folder(sdir):
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


# def sort_tracks(tracks: Union[list[str], list[Path]]) -> Union[list[str],list[Path]]:
def sort_tracks(tracks: list[Path]) -> list[Path]:
    assert is_instance(tracks, list[Path])

    def _split(name) -> tuple[int, int]:
        parts = name.split("_")
        return int(parts[0]), int(parts[1])

    if len(tracks) == 0:
        return tracks

    if isinstance(tracks[0], Path):
        is_path = True
        track_names = [
            track_dir.name
            for track_dir in tracks
        ]
    else:
        is_path = False
        track_names = tracks

    track_names = sort_by_key(track_names, key=_split)
    if is_path:
        tracks_root = tracks[0].parent
        sorted_tracks = [
            (tracks_root / track_name)
            for track_name in track_names
        ]
    else:
        sorted_tracks = track_names
    return sorted_tracks
# end

# ---------------------------------------------------------------------------

def info_violation(
    meta_records_map: dict[CAM_TRACK_NAME, RECORD],
    this_to_combine: list[CAM_TRACK_NAME],
    violation: str
) -> tuple[bool, Optional[str]]:
    has_violation = any(meta_records_map[track_id].get(violation, False) for track_id in this_to_combine)
    if not has_violation:
        return has_violation, None

    img_violation = "img_" + violation
    imgs_violation = [meta_records_map[track_id].get(img_violation, None) for track_id in this_to_combine if
                           meta_records_map[track_id].get(img_violation, None) is not None]

    return has_violation, imgs_violation[0] if len(imgs_violation) > 0 else None


