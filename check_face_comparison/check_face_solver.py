import warnings
import logging.config
from pathlib import Path

from joblib import Parallel, delayed

from post_processing.face_name_solver import FaceNameSolver
from post_processing.utils import select_tracks_roots, init_modules
from stdlib.jsonx import JSONConfiguration



def analyze_par(tracks_root: Path, dbname: str, tid: int, n: int):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")

    CONFIG = JSONConfiguration.load("config/config_post_dev.json")

    init_modules()

    fns = FaceNameSolver(CONFIG, f":{tid}")
    # fs.analyze(tracks_root, f"/dbf{tid}")
    fns.analyze(tracks_root, dbname)
    pass
# end


def main():

    roots = [
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_2026_flat_1"),
        Path(r"D:\Projects.ebtic.datasets\lab_monitoring_data\tmp_result_2026_flat_2")
    ]

    for root in roots:
        tracks_roots = select_tracks_roots(root)
        n = len(tracks_roots)

        # Parallel(n_jobs=6)(
        #     delayed(analyze_par)(tracks_root, "dbo", i, n)
        #     for i, tracks_root in enumerate(tracks_roots)
        # )

        for i, tracks_root in enumerate(tracks_roots):
            analyze_par(tracks_root, "dbo", i, n)
        # end
    # end
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")
    main()

