import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

import stdlib.loggingx as logging
from human.clipreid import ClipReID
from stdlib.is_instance import is_instance
from stdlib.jsonx import JSONConfiguration
from stdlib.qname import create_from
from .utils import EMBEDDING, LabMonitoring, METRIC_TYPES

TOP_SIMILARITY = 0.9999


class FaceSolver(LabMonitoring):
    def __init__(self, CONFIG: JSONConfiguration):
        super().__init__(CONFIG, "face_solver")

        self.enabled: bool = CONFIG.get("face_solver.enabled", False)
        self.faces_store: Path = Path(CONFIG.get("face_solver.faces_store", ".faces_store"))

        self.model_name: str = CONFIG.get("face_solver.model_name", "DukeMTMC__cnn_base")
        self.metric: str = CONFIG.get("face_solver.distance_metric", "cosine")
        self.linkage: str = CONFIG.get("face_solver.linkage", "complete")
        self.similarity_threshold: float = CONFIG.get("face_solver.similarity_threshold", 0.5)

        assert is_instance(self.metric, METRIC_TYPES)
        assert is_instance(self.linkage, Literal["average", "complete", "single", "ward"])

        self.embedding = create_from(CONFIG.get("face_solver.embedding"))
        # self.faces_store.mkdir(parents=True, exist_ok=True)

        self._faces_database: dict[str, list[EMBEDDING]] = {}
        self._means_database: dict[str, list[EMBEDDING]] = {}

        self._log = logging.getLogger("FaceSolver")

    def analyze(self, tracks_root: Path, face_db_suffix=""):
        assert is_instance(tracks_root, Path)
        assert is_instance(face_db_suffix, str)

        self._log.info(f"Analyzing {tracks_root} ...")

        self._track_root = tracks_root

        # create the face_db (a directory)
        if len(face_db_suffix) > 0:
            self.faces_store = self.faces_store.parent / (self.faces_store.name + f"_{face_db_suffix}")
        self.faces_store.mkdir(parents=True, exist_ok=True)

        # preload the faces database
        self._load_faces_database()

        # scan the tracks
        trace_dirs = [
            track_dir
            for track_dir in tracks_root.iterdir()
            if self._is_track_valid(track_dir)
        ]
        n = len(trace_dirs)

        for i, track_dir in enumerate(trace_dirs):
            face_dir = track_dir / "face"
            if not face_dir.is_dir(): continue

            self._log.infot(f"... {track_dir.name} ({i+1:4}/{n})")

            faces_embeddings: list[EMBEDDING] = self._get_images_embedding(face_dir)

            person_name, similarity = self._find_person(faces_embeddings)

            if similarity <= self.similarity_threshold:
                person_name = self._create_new_person(face_dir, faces_embeddings)
                self._log.info(f"... ... created new person: {person_name} (similarity={similarity:.3})")
            elif similarity >= TOP_SIMILARITY:
                self._log.infot(f"... ... already registered: {person_name}")
                pass
            else:
                self._log.infot(f"... ... update person: {person_name} (similarity={similarity:.3})")
                self._update_person(person_name, face_dir, faces_embeddings)
            pass
        # end
        self._log.info(f"Done")
        self._cleanup()
    # end

    def _is_track_valid(self, track_dir: Path):
        # if the track name is '<camid>_<trackid>_DONE'
        if not track_dir.name.endswith("_DONE"):
            return False

        # random_crop must exist
        if not (track_dir / "random_crop").exists():
            return False

        # face directory must exist
        if not (track_dir / "face").exists():
            return False

        return True

    def _load_faces_database(self):
        for face_dir in self.faces_store.iterdir():
            if face_dir.is_file(): continue
            face_embeddings = self._get_images_embedding(face_dir)
            self._faces_database[face_dir.name] = face_embeddings

    def _get_images_embedding(self, image_dir: Path):
        images_embedding: list[EMBEDDING] = []
        for image_file in image_dir.iterdir():
            # embedding: EMBEDDING = ClipReID.represent(image_file, self.model_name)
            embedding: EMBEDDING = self.embedding.embedding(image_file)
            images_embedding.append(embedding)
        return images_embedding

    def _find_person(self, face_embeddings: list[EMBEDDING]) -> tuple[str, float]:
        best_name = "unknown"
        best_similarity = 0.
        for name in self._faces_database:
            if self.linkage == "complete":
                dist_matrix = cdist(face_embeddings, self._faces_database[name], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.max()
            elif self.linkage == "average":
                dist_matrix = cdist(face_embeddings, self._means_database[name], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.max()
            elif self.linkage == "single":
                dist_matrix = cdist(face_embeddings, self._faces_database[name], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.min()
            else:
                raise ValueError(f"Unknown linkage: {self.linkage}")

            # sim_matrix = 1 - dist_matrix
            # similarity = sim_matrix.max()

            if similarity > best_similarity:
                best_name = name
                best_similarity = similarity
        # end
        return best_name, best_similarity
    # end

    def _create_new_person(self, face_dir: Path, faces_embeddings: list[EMBEDDING]) -> str:
        last_person_id = self._find_last_person_id()
        person_name = f"Person-{last_person_id+1}"

        person_dir = self.faces_store / person_name
        shutil.copytree(str(face_dir), str(person_dir))

        # faces_embeddings = self._get_images_embedding(person_dir)
        means_embeddings = [np.array(faces_embeddings).mean(axis=0)]

        self._faces_database[person_name] = faces_embeddings
        self._means_database[person_name] = means_embeddings

        return person_name
    # end

    def _update_person(self, person_name: str, face_dir: Path, update_embeddings: list[EMBEDDING]):
        person_dir = self.faces_store / person_name
        shutil.copytree(str(face_dir), str(person_dir), dirs_exist_ok=True)

        # faces_embeddings = self._get_images_embedding(person_dir)
        faces_embeddings = self._faces_database[person_name] + update_embeddings
        means_embeddings = [np.array(faces_embeddings).mean(axis=0)]

        self._faces_database[person_name] = faces_embeddings
        self._means_database[person_name] = means_embeddings
        pass

    def _find_last_person_id(self) -> int:
        last_person_id = 0
        for face_name in self._faces_database:
            if not face_name.startswith("Person-"): continue
            person_id = int(face_name[len("Person-"):])
            if person_id > last_person_id:
                last_person_id = person_id
        return last_person_id

    def _cleanup(self):
        pass
# end
