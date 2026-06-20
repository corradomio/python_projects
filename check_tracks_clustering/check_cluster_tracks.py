from collections import defaultdict
from pathlib import Path
import stdlib.loggingx as logging
from stdlib import jsonx
from stdlib import csvx

from cluster_tracks import ClusterTracks
from stdlib.jsonx import JSONConfiguration

# TRACKS_ROOT: Path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260513_155653")
# TRACKS_ROOT: Path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260609_105432")
TRACKS_ROOT: Path = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-08\20260513_155653")



def has_face(track_dir: Path) -> bool:
    face_dir = track_dir / "face"
    return face_dir.exists()


def save_cluster_id(track_dir: Path, cluster_id: int, similarity: float):
    cluster_info = {
        "cluster_id": cluster_id,
        "similarity": similarity
    }
    # cluster_path = track_dir / f"cluster_{track_id}.json"
    # jsonx.dump(cluster_info, cluster_path)

    cluster_path = track_dir / f"random_crop/cluster_{cluster_id}.json"
    jsonx.dump(cluster_info, cluster_path)
    pass


# ---------------------------------------------------------------------------

def main():
    log = logging.getLogger("main")

    CONFIG = JSONConfiguration.load("config_post_dev.json")

    ct = ClusterTracks(CONFIG)

    ct.analyze(TRACKS_ROOT)

    cluster_dict = ct.clusters_tracks

    # log.info(f"Processing {TRACKS_ROOT}")
    # for track_dir, _, _ in TRACKS_ROOT.walk():
    #     if not track_dir.name.endswith("_DONE"):
    #         continue
    #
    #     log.infot(f"... {track_dir.parent.parent.name}/{track_dir.parent.name}/{track_dir.name}")
    #
    #     random_crop = track_dir / "random_crop"
    #     if not random_crop.exists():
    #         continue
    #
    #     cluster_id, similarity, comparisons = ct.find_cluster_id(track_dir)
    #
    #     save_cluster_id(track_dir, cluster_id, similarity)
    #
    #     cluster_dict[cluster_id].append(track_dir.name)
    # pass

    cluster_list = [
        [f"C{cluster_id}", ",".join(cluster_dict[cluster_id])]
        for cluster_id in cluster_dict
    ]

    csvx.dump(cluster_list, f"clusters_{TRACKS_ROOT.name}.csv", header=["Cluster","Tracks"])
# end


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    main()
