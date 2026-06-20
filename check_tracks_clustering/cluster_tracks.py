import shutil
from pathlib import Path

import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist

from stdlib import loggingx as logging
from stdlib.sortedx import sort_by_key
from human.clipreid import ClipReID
from stdlib.is_instance import is_instance
from stdlib import jsonx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING = np.ndarray

IMAGES_EMBEDDING = list[EMBEDDING]

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# ClusterTracks
# ---------------------------------------------------------------------------

class ClusterTracks:

    def __init__(
        self,
        CONFIG: jsonx.JSONConfiguration
    ):
        self.CONFIG = CONFIG
        # store_root: str | Path=Path(r".clusters_tracks"),
        # images_folders: list[str]=["random_crop", "face_recognition", "face_recognition_fallback"],
        # threshold: float = 0.85,
        # top_k: int = 1,
        # model_name: str = "MSMT17__cnn_clipreid"

        # where to save the image samples for the tracks in a cluster
        self.store_root: Path = Path(CONFIG.get("cluster_tracks.store_root", ".clusters_store"))
        # minimum distance between a track and a cluster
        self.threshold: float = CONFIG.get("cluster_tracks.threshold", 0.92)
        # how many images to analyze to compute the distance:
        #   0) similarity_matrix.mean()
        #   1) similarity_matrix.max()
        #  >1) mean of top k similarity measures
        #  -1) similarity distance between embedding means
        self.top_k: int = CONFIG.get("cluster_tracks.top_k", -1)
        # ReID model to use, for now based on ClipReID
        self.model_name: str = CONFIG.get("cluster_tracks.model_name",  "MSMT17__cnn_clipreid")
        # list of image folders where to collect the images
        self.images_folders: list[str] = CONFIG.get("cluster_tracks.image_folders",  ["random_crop"])
        # minimum track length (in seconds)
        self.min_duration: int = CONFIG.get("cluster_tracks.min_duration", 1)
        # list of folders MUST BE PRESENT to process the track
        self.validate_folders: list[str] = CONFIG.get("cluster_tracks.validate_folders",  ["random_crop"])
        # if to include tracks with 'more_persons=true'
        self.more_persons: bool = CONFIG.get("cluster_tracks.more_persons",  False)
        # if to save the track samples store_root
        self.save_track: bool = CONFIG.get("cluster_tracks.save_track",  True)
        # if to save a JSON file containing information onf the selected cluster in the track_dir/random_crop
        self.save_images: bool = CONFIG.get("cluster_tracks.save_images",  False)

        self._clusters_embeddings: dict[int, list[list[EMBEDDING]]] = {}
        self._clusters_tracks: dict[int, list[str]]= {}
        self._log = logging.getLogger("TracksDatabase")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def clusters_tracks(self) -> dict[int, list[str]]:
        return self._clusters_tracks

    # -----------------------------------------------------------------------
    # analyze
    # -----------------------------------------------------------------------

    def analyze(self, tracks_root: str|Path):

        def _split(name: str) -> tuple[int, int]:
            parts = name.split("_")
            return int(parts[0]), int(parts[1])

        if isinstance(tracks_root, str):
            tracks_root = Path(tracks_root)

        self._log.info(f"Analyzing {tracks_root} ...")

        # 1) collect the list of track names to process
        track_names = [
            track_dir.name
            for track_dir in tracks_root.iterdir()
            if track_dir.name.endswith("_DONE")
        ]
        # 2) sort track names lexicographically
        track_names = sort_by_key(track_names, key=_split)
        n = len(track_names)
        # 3) recreate the track dirs to process
        tracks_dirs = [
            tracks_root / track_name
            for track_name in track_names
        ]

        for i, track_dir in enumerate(tracks_dirs):
            # if not track_dir.name.endswith("_DONE"):
            #     continue

            if not self._is_track_valid(track_dir):
                continue

            self._log.infot(f"... {track_dir.name} ({i+1}/{n})")

            # skip tracks without "random_crop"
            random_crop = track_dir / "random_crop"
            if not random_crop.exists():
                continue

            cluster_id, similarity, comparisons = self.find_cluster_id(track_dir)

            self._save_track(track_dir, cluster_id, similarity)
        # end
        self._log.info(f"Done")
    # end

    def _is_track_valid(self, track_dir: Path):
        # the if the track name is '<camid>_<trackid>_DONE'
        if not track_dir.name.endswith("_DONE"):
            return False

        # check if all folders in 'valid_folders' are present
        for valid_name in self.validate_folders:
            if not (track_dir / valid_name).exists():
                return False

        # check if 'more-Persons' is accepted
        if self.more_persons and self.min_duration == 0:
            return True

        # check if the track duration is greater than 'min_duration'
        meta = jsonx.load(track_dir / "meta.json")
        track_duration = self._track_duration(meta)
        if track_duration < self.min_duration:
            return False

        # ensure the track has 'more_persons=false'
        more_persons = meta["more_persons"]
        return self.more_persons or not more_persons

    def _track_duration(self, meta):
        start_time: datetime = datetime.strptime(meta["present_start"], DATETIME_FORMAT)
        end_time: datetime = datetime.strptime(meta["present_end"], DATETIME_FORMAT)
        return (end_time - start_time).seconds


    def _save_track(self, track_dir: Path, cluster_id: int, similarity: float):
        if not self.save_track:
            return

        cluster_info = {
            "cluster_id": cluster_id,
            "similarity": similarity
        }
        # cluster_path = track_dir / f"cluster_{track_id}.json"
        # jsonx.dump(cluster_info, cluster_path)

        cluster_path = track_dir / f"random_crop/cluster_{cluster_id}.json"
        jsonx.dump(cluster_info, cluster_path)
        pass

    # -----------------------------------------------------------------------
    # find_cluster_id
    # -----------------------------------------------------------------------

    def find_cluster_id(self, track_dir: Path) -> tuple[int, float, list[tuple[str, float]]]:
        # return: cluster_di, similarity, list[tuple[list, similarity]]

        if is_instance(track_dir, str):
            track_dir = Path(track_dir)

        # Note: embedding already prepared for top_k == -1
        images_embedding = self._get_images_embedding(track_dir)

        best_similarity = 0
        best_cluster_id = -1

        for cluster_id in self._clusters_embeddings:
            similarity = self._compare_against_cluster(images_embedding, cluster_id)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        # end

        if best_similarity >= self.threshold:
            self._clusters_embeddings[best_cluster_id] += [images_embedding]
            self._clusters_tracks[best_cluster_id].append(track_dir.name)
        else:
            best_cluster_id = len(self._clusters_embeddings) + 1
            self._clusters_embeddings[best_cluster_id] = [images_embedding]
            self._clusters_tracks[best_cluster_id]= [track_dir.name]
        # end

        self._save_images(best_cluster_id, track_dir)
        return best_cluster_id, best_similarity, []

    def _compare_against_cluster(self, images_embedding:IMAGES_EMBEDDING, cluster_id: int):
        cluster_embeddings = self._clusters_embeddings[cluster_id]

        cluster_similarity = 0
        for cluster_embedding in cluster_embeddings:
            similarity = self._pair_similarity(images_embedding, cluster_embedding)
            if similarity > cluster_similarity:
                cluster_similarity = similarity
        return cluster_similarity

    def _pair_similarity(self, embeddings_i: IMAGES_EMBEDDING, embeddings_j: IMAGES_EMBEDDING):
        # WARNING: similarity is in range [0,1]
        #   WITH 1 THE BEST  SIMILARITY
        #   WITH 0 THE WORST SIMILARITY
        assert is_instance(embeddings_i, IMAGES_EMBEDDING)
        assert is_instance(embeddings_j, IMAGES_EMBEDDING)

        if len(embeddings_i) == 0 or len(embeddings_j) == 0:
            return 0.

        dist_matrix = cdist(embeddings_i, embeddings_j, metric='cosine')
        similarity_matrix = 1 - dist_matrix

        if self.top_k == 1:
            pair_similarity = similarity_matrix.max()
        elif self.top_k == 0:
            pair_similarity = similarity_matrix.mean()
        elif self.top_k > 1:
            similarity_vector = similarity_matrix.reshape(-1)
            # sorted ascending
            similarity_vector.sort()
            k = min(self.top_k, len(similarity_vector))
            pair_similarity = similarity_vector[-k:].mean()
        elif self.top_k == -1:
            # Note: ALREADY prepared
            # mean_embeddings_i = np.array(embeddings_i).mean(axis=0).reshape((1, -1))
            # mean_embeddings_j = np.array(embeddings_j).mean(axis=0).reshape((1, -1))
            # dist_matrix = cdist(mean_embeddings_i, mean_embeddings_j, metric='cosine')
            # pair_similarity = (1 - dist_matrix).max()
            pair_similarity = similarity_matrix.max()
        else:
            raise ValueError(f"Unsupported top_k {self.top_k}")
        return pair_similarity

    def _get_images_embedding(self, track_dir: Path) -> list[EMBEDDING]:
        images_embedding: list[EMBEDDING] = []

        for images_folder in self.images_folders:
            images_dir = track_dir / images_folder
            if not images_dir.exists():
                continue

            for image_file in images_dir.iterdir():
                if image_file.suffix not in [".png", ".jpg"]:
                    continue

                embedding: EMBEDDING = ClipReID.represent(image_file, self.model_name)
                images_embedding.append(embedding)

        # prepare the embedding to speedup the computation
        if self.top_k == -1:
            images_embedding = [np.array(images_embedding).mean(axis=0).reshape(-1)]

        return images_embedding

    def _save_images(self, cluster_id: int, track_dir: Path):
        if not self.save_images:
            return

        data_name = track_dir.parent.name
        images_dir = track_dir / "random_crop"

        track_name = images_dir.parent.name
        cluster_path: Path = self.store_root / f"{data_name}/cluster_{cluster_id}/{track_name}"
        cluster_path.mkdir(parents=True, exist_ok=True)

        for image in images_dir.iterdir():
            shutil.copy(image, cluster_path / image.name)
            break
        pass
    # end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------



