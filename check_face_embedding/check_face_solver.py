import logging.config
from pathlib import Path
from stdlib.jsonx import JSONConfiguration
from post_processing.face_solver import FaceSolver
from post_processing.face_analyzer import FaceAnalyzer


def main():

    CONFIG = JSONConfiguration.load("config/config_post_dev.json")

    # -- in office --

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-26-13-30")
    # faces_suffix = "26.1"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-26-23-45")
    # faces_suffix = "26.2"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-27-13-30")
    # faces_suffix = "27.1"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-27-23-45")
    # faces_suffix = "27.2"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-28-13-30")
    # faces_suffix = "28.1"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-28-23-45")
    # faces_suffix = "28.2"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-29-08-36")
    # faces_suffix = "29.1"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-30-00-00")
    # faces_suffix = "30.1"

    # -- at home --

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\20260624_100020\20260623_152000")
    # faces_suffix = "24.1"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-22\20260609_105432")
    # faces_suffix = "22.1"

    tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-08\20260513_155653")
    faces_suffix = "08.1"

    # -- end --

    fa = FaceAnalyzer(CONFIG)
    fa.analyze(tracks_root)

    fs = FaceSolver(CONFIG)
    fs.analyze(tracks_root, faces_suffix)
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    main()

