from pathlib import Path
from stdlib import loggingx as logging
from stdlib.jsonx import  JSONConfiguration
from post_processing.face_extractor import FaceExtractor
from joblib import Parallel, delayed


def extract_faces(root: Path):
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    log.info(f"... {root}")

    CONFIG = JSONConfiguration.load("config_post_dev.json")
    fe = FaceExtractor(CONFIG)
    fe.extract_faces(root)
# end


# ROOT = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260513_155653")
ROOT = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260609_105432")


def main():
    N_JOBS = 4

    print("Start ...")

    Parallel(n_jobs=N_JOBS)(delayed(extract_faces)(root) for root in ROOT.iterdir())

    # for root in ROOT.iterdir():
    #     extract_faces(root)

    print("Done")
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    log.info("Started ...")
    main()
