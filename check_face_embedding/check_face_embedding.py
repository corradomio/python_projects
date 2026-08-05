import logging.config
from pathlib import Path
from stdlib.jsonx import JSONConfiguration
from post_processing.face_solver import FaceSolver
from post_processing.face_analyzer import FaceAnalyzer
from human.arcface import ArcFace, ARCFACE_MODEL_NAMES
from stdlib.tprint import tprint


def main():
    roots = [
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_2026_flat_2"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_2026_flat_1"),
    ]

    for root in roots:
        for tracks_dirs in root.iterdir():
            if not tracks_dirs.is_dir(): continue

            for track_dir in tracks_dirs.iterdir():
                if not track_dir.is_dir(): continue

                face_dir = track_dir / "face"
                if not face_dir.exists(): continue

                for face_file in face_dir.iterdir():
                    if not face_file.name.endswith(".jpg"): continue

                    for model_name in ARCFACE_MODEL_NAMES:
                        tprint(face_file, force=False)
                        emb = ArcFace.represent(face_file, model_name)
                        if emb is None or len(emb) == 0:
                            tprint(f"... [{model_name}] {face_file} invalid")
                            pass
                        pass



if __name__ == "__main__":
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    main()
