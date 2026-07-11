import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging.config
from pathlib import Path

from post_processing.image_classifier import ImageClassifier
from post_processing.image_embedding import ImageEmbedding
from post_processing.pose_analyzer import PoseAnalyzer
from stdlib.jsonx import JSONConfiguration
from post_processing.face_solver import FaceSolver
from post_processing.face_analyzer import FaceAnalyzer


def main():

    CONFIG = JSONConfiguration.load("config/config_post_dev.json")

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-29-08-36")
    # db_suffix = "/29.1"

    tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-06-30-00-00")
    db_suffix = "/30.1"

    # tracks_root = Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result\2026-07-05-23-45")
    # db_suffix = "/05.2"

    fa = FaceAnalyzer(CONFIG)
    pa = PoseAnalyzer(CONFIG)
    fs = FaceSolver(CONFIG)
    ie = ImageEmbedding(CONFIG, fa, pa)
    ic = ImageClassifier(CONFIG, ie)

    fa.analyze(tracks_root)
    pa.analyze(tracks_root)
    fs.analyze(tracks_root, db_suffix)
    ie.analyze(tracks_root)
    ic.analyze(tracks_root, db_suffix)
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    main()

