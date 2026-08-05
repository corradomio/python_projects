import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Union, Optional, Iterable, Sized
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

DAY = str
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
JSON = dict

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

VIOLATIONS = [
    'not_dress_well',
    'not_glove_well',
    'unauthorised_access',
    'unauthorised_operation_A',
    'unauthorised_machine_touching_B',
    'cleaner_or_security'
]

# MANDATORY_BOOLEAN_FIELDS = VIOLATIONS + [
#
# ]

# MANDATORY_STRING_FIELDS = [
#     'face',
#     'random'
#     'img_not_dress_well',
#     'img_not_glove_well',
#     'img_unauthorised_access',
#     'img_unauthorised_operation_A',
#     'img_unauthorised_machine_touching_B',
#     'img_cleaner_or_security',
#
#     "authorised_access",
#     "authorised_machine_A",
#     "authorised_machine_B"
# ]


NOT_ASSIGNED_NAMES = ["face_not_in_DB", "NO_FACES_SAVED", "KU_FACE_RECOGNITION_UNAVAILABLE"]

METRIC_TYPES = Literal[
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
    "hamming", "jaccard", "jensenshannon", "mahalanobis", "matching", "minkowski", "rogerstanimoto",
    "russellrao", "seuclidean", "sokalsneath", "sqeuclidean", "yule"
]

LINKAGE_TYPE = Literal["average", "complete", "single", "ward", "centroid"]

SIMULATED_VIOLATION_TYPE = Literal["unauthorised_access", "unauthorised_operation_A", "unauthorised_machine_touching_B"]

# ---------------------------------------------------------------------------
# LabMonitoring
# ---------------------------------------------------------------------------

# Global registry of created modules.
# Each module MUST HAVE a unique name
LAB_MONITORING_MODULES: dict[str, "LabMonitoring"] = {}


class LabMonitoring:
    def __init__(self, CONFIG: JSONConfiguration, module: str, register_name: Optional[str]=None):
        global LAB_MONITORING_MODULES

        assert isinstance(CONFIG, JSONConfiguration)
        assert isinstance(module, str)

        if register_name is None: register_name = module

        self.CONFIG = CONFIG
        self.module = module

        assert module in CONFIG, f"Missing configuration for '{module}'"

        assert module not in LAB_MONITORING_MODULES, f"Module {module} already registered. Missing 'init_modules()'?"
        LAB_MONITORING_MODULES[register_name] = self
        pass

    def get(self, config_key, default="raise"):
        return self.CONFIG.get(f"{self.module}.{config_key}", default=default)

    def _is_track_valid(self, track_dir: Path) -> bool:
        if not track_dir.name.endswith("_DONE"):
            return False

        if not (track_dir / "meta.json").exists():
            return False

        if not (track_dir / "random_crop").exists():
            return False

        return True
    # end


def get_module(module: str) -> LabMonitoring:
    assert isinstance(module, str)
    assert module in LAB_MONITORING_MODULES, f"Module {module} not loaded yet"
    return LAB_MONITORING_MODULES[module]


def init_modules():
    global LAB_MONITORING_MODULES
    LAB_MONITORING_MODULES = {}


# ---------------------------------------------------------------------------
# DataServer
# ---------------------------------------------------------------------------

# class DataServer(LabMonitoring):
#     def __init__(self, CONFIG: JSONConfiguration, component: str):
#         super().__init__(CONFIG, component)
#
#     def get_images_to_transfer(
#         self,
#         tracks_root: Path,
#         meta_records_map: dict[CAM_TRACK_NAME, RECORD],
#         date_in_id: str,
#         to_combine: list[list[CAM_TRACK_NAME]]
#     ) -> dict[str, str]:
#         ...
#
#     def save_data(
#         self,
#         tracks_root: Path,
#         meta_records_map: dict[CAM_TRACK_NAME, RECORD],
#         date_in_id: str,
#         to_combine: list[list[CAM_TRACK_NAME]]
#     ) -> dict[str, str]:
#         ...


# ---------------------------------------------------------------------------
# FaceServer
# ---------------------------------------------------------------------------

class FaceServer(LabMonitoring):

    def solve_faces_name(
        self,
        tracks_root: Path,
        track_names: list[CAM_TRACK_NAME],
        track_meta_records: list[RECORD]
    ):
        ...


# ---------------------------------------------------------------------------
# TracksCluster
# ---------------------------------------------------------------------------

class TracksCluster(Sized, Iterable):
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

    @property
    def tracks_list(self) -> list[CAM_TRACK_NAME]:
        return sort_track_names(self.track_names)

    def __len__(self):
        return len(self.track_names)

    def __iter__(self):
        return self.track_names.__iter__()
# end


class ClusterTracks(LabMonitoring):
    def __init__(self, CONFIG: JSONConfiguration, component: str):
        super().__init__(CONFIG, component)

    def track_cluster_map(self) -> dict[CAM_TRACK_NAME, CLUSTER_ID]:
        ...

    def cluster_tracks_map(self) -> dict[CLUSTER_ID, TracksCluster]:
        ...

    def has_track(self, track_name: str) -> bool:
        ...

    def analyze(self, tracks_root: Path, tracks_meta_map: dict[CAM_TRACK_NAME, RECORD], save: bool = True) -> bool:
        ...
# end


# ---------------------------------------------------------------------------
# Numerical Utilities
# ---------------------------------------------------------------------------

BASE64_PLAIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
BASE64_URL   = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
BASE64_HASH  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
BASE64 = len(BASE64_HASH)

def to64(v: int|float) -> str:
    return toBase(v, 64)

def to36(v: int|float) -> str:
    return toBase(v, 36)

def toBase(v: int|float, b:int=36) -> str:
    v = int(v)
    if v == 0: return "0"
    c = ""
    while v != 0:
        d = v % b
        v //= b
        c = BASE64_URL[d:d + 1] + c
    # end
    return c


def chop(x, xmax: int):
    if x < 0:
        x = 0
    elif x > xmax:
        x = xmax
    return int(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def select_folders_to_process(from_path: Path) -> list[Path]:
    if _is_valid_folder(from_path):
        return [from_path]

    # folders_to_process = [
    #     sub_folder
    #     for sub_folder in from_path.iterdir()
    #     if _is_valid_folder(sub_folder)
    # ]

    folders_to_process = []
    for tracks_root, dirs, files in from_path.walk():
        if _is_valid_folder(tracks_root):
            folders_to_process.append(tracks_root)
            dirs.clear()
            files.clear()
    pass
    return folders_to_process
# end


def find_tracks_start(tracks_root: Path, single=False) -> datetime:
    tracks_start = datetime.now()
    for tracks_dir in tracks_root.iterdir():
        if not tracks_dir.name.endswith("_DONE"): continue
        meta_file = tracks_dir / "meta.json"
        if not meta_file.exists(): continue
        meta = jsonx.load(meta_file)
        present_start = datetime.strptime(meta["present_start"], DATETIME_FORMAT)
        if present_start < tracks_start:
            tracks_start = present_start
        if single: break
    pass
    return tracks_start
# end


def _is_valid_folder(tracks_dir: Path):
    if not is_timestamp_folder(tracks_dir):
        return False

    # Note: "impurity.json" and "segmented.json" are present
    # ONLY IF the folder is PRE-processed

    # the folder MUST contain subfolders (NOT only files)
    if is_root_folder_empty(tracks_dir):
        return False
    # the folder MUST BE NOT impure (containing more days)
    if (tracks_dir / "impurity.json").exists():
        return False
    # the folder MUST BE NOT A day segment (part of the day)
    if (tracks_dir / "segmented.json").exists():
        return False

    # Skip the TODAY tracks
    # today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # tracks_start = _find_tracks_start(tracks_dir, True)
    # if tracks_start >= today:
    #     return False
    return True
# end

# ---------------------------------------------------------------------------

def has_tracks(dirnames: list[str]):
    for dirname in dirnames:
        if dirname.endswith('_DONE'):
            return True
    return False


def select_tracks_roots(scan_root: Path, seg_imp=True):
    tracks_roots = []
    for tracks_root in scan_root.iterdir():
        if not tracks_root.is_dir():
            continue

        # if seg_imp and ((tracks_root / "impurity.json").exists() or (tracks_root / "segmented.json").exists()):
        #         continue
        if (tracks_root / "impurity.json").exists():
            continue
        if (tracks_root / "segmented.json").exists():
            continue
        # this name is used to merge the tracks
        # if tracks_root.name.endswith("-00-00"):
        #     continue
        if tracks_root.name.endswith("_DONE"):
            continue

        # if has_tracks(dirnames):
        #     tracks_roots.append(tracks_root)
        #     dirnames.clear()
        #     filenames.clear()

        tracks_roots.append(tracks_root)
    # end
    return tracks_roots


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
    if not meta_file.exists():
        return {}

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
    check if the name of the folder is <YYYYMMDD_hhmmss> OR
        <YYYY-MM-DD>
        <YYYY-MM-DD-HH>
        <YYYY-MM-DD-HH-MM>
        <YYYY-MM-DD-HH-MM-SS>
    :param folder:
    :return:
    """
    assert isinstance(folder, Path)
    if not folder.is_dir(): return False

    name = folder.name

    if name.endswith("_DONE"):
        return False

    # <YYYYMMDD_hhmmss>
    if "_" in name:
        parts = name.split('_')
        if len(parts) != 2:
            return False
        if len(parts[0]) != 8 or len(parts[1]) != 6:
            return False
        return True

    # <YYYY-MM-DD>
    # <YYYY-MM-DD-HH>
    # <YYYY-MM-DD-HH-MM>
    # <YYYY-MM-DD-HH-MM-SS>
    if "-" in name:
        parts = name.split('-')
        if len(parts) < 3:
            return False
        if len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
            return False
        if len(parts) >= 4 and len(parts[3]) != 2:
            return False
        if len(parts) >= 5 and len(parts[4]) != 2:
            return False
        if len(parts) >= 6 and len(parts[5]) != 2:
            return False
        return True

    return False


def is_root_folder_empty(root_folder: Path) -> bool:
    if not root_folder.exists():
        return True
    for sdir in root_folder.iterdir():
        if is_timestamp_folder(sdir):
            continue
        if sdir.is_file():
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
# end


# def sort_tracks(tracks: list[Path]|list[str]) -> list[Path]|list[str]:
#     assert is_instance(tracks, list[Path])
#
#     if len(tracks) == 0:
#         return tracks
#
#     if isinstance(tracks[0], Path):
#         is_path = True
#         track_names = [
#             track_dir.name
#             for track_dir in tracks
#         ]
#     else:
#         is_path = False
#         track_names = tracks
#
#     track_names = sort_by_key(track_names, key=_split)
#
#     if is_path:
#         tracks_root = tracks[0].parent
#         sorted_tracks = [
#             (tracks_root / track_name)
#             for track_name in track_names
#         ]
#     else:
#         sorted_tracks = track_names
#     return sorted_tracks
# # end


def sort_tracks(track_dirs: list[Path]) -> list[Path]:
    assert is_instance(track_dirs, list[Path])

    if len(track_dirs) == 0:
        return []

    tracks_root = track_dirs[0].parent

    track_names = [
        track_dir.name
        for track_dir in track_dirs
    ]

    track_names = sort_track_names(track_names)

    track_dirs = [
        (tracks_root / track_name)
        for track_name in track_names
    ]

    return track_dirs
# end


def sort_track_names(track_names: list[str]) -> list[str]:
    assert is_instance(track_names, list[str])

    def _split(name) -> tuple[int, int]:
        parts = name.split("_")
        return int(parts[0]), int(parts[1])

    if len(track_names) == 0:
        return []

    track_names = sort_by_key(track_names, key=_split)

    return track_names
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


