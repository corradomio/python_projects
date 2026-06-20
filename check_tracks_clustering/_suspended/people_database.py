import logging
import numpy as np
import shutil
from pathlib import Path

from scipy.spatial.distance import cdist

from stdlib.is_instance import is_instance
from reidentx.clipreid import ClipReID


EMBEDDING = np.ndarray
IMAGES_EMBEDDING = list[EMBEDDING]


#
#   <people_database_root>
#       <id>
#           <image1>.png
#           ...

class PeopleDatabase:
    def __init__(
        self,
        root: str | Path=Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\.people_database"),
        threshold: float=0.85,
        top_k:int = 1,
        model_name: str="MSMT17__cnn_clipreid"
    ):
        if isinstance(root, str):
            root = Path(root)

        assert root.exists()

        self.root: Path = root
        self.threshold: float = threshold
        self.top_k: int = top_k
        self.model_name: str = model_name

        self._next_id: int = 0
        self._scan_for_next_id()

        self._people_embeddings: dict[int, list[EMBEDDING]] = {}
        self._tracks_embedding: dict[str, list[EMBEDDING]] = {}
        self._log = logging.getLogger("PeopleDatabase")

    def _scan_for_next_id(self):
        # scan 'root' to find which is the NEXT VALID id to use
        self._next_id = 1
        for dir in self.root.iterdir():
            dir_id = int(dir.name)
            if dir_id >= self._next_id:
                self._next_id = dir_id+1

    def find_people_id(self, images_dirs: list[str|Path]) -> tuple[int, float, bool]:
        assert is_instance(images_dirs, list[str|Path])
        images_dirs = [
            (Path(images_dir) if isinstance(images_dir, str) else images_dir)
            for images_dir in images_dirs
        ]

        # scan the list of peoples to
        images_embedding = self._get_images_embedding(images_dirs)

        best_similarity = 0
        best_person_id = -1

        for person_id in range(self._next_id):
            person_embeddings = self._get_person_embeddings(person_id)

            similarity = self.pair_similarity(images_embedding, person_embeddings)
            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person_id
            # end
        # end
        return best_person_id, best_similarity, best_similarity>= self.threshold

    def pair_similarity(self, embeddings_i: IMAGES_EMBEDDING, embeddings_j: IMAGES_EMBEDDING):
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
            mean_embeddings_i = np.array(embeddings_i).mean(axis=0).reshape((1, -1))
            mean_embeddings_j = np.array(embeddings_j).mean(axis=0).reshape((1, -1))
            dist_matrix = cdist(mean_embeddings_i, mean_embeddings_j, metric='cosine')
            pair_similarity = (1 - dist_matrix).max()
        else:
            raise ValueError(f"Unsupported top_k {self.top_k}")

        return pair_similarity
    # end

    def create_person_id(self, images_dirs: list[str|Path]) -> tuple[int, Path]:
        assert is_instance(images_dirs, list[str | Path])
        images_dirs = [
            (Path(images_dir) if isinstance(images_dir, str) else images_dir)
            for images_dir in images_dirs
        ]

        person_id = self._next_id
        self._next_id += 1

        person_dir: Path = self.root / str(person_id)
        person_dir.mkdir()

        saved = False

        for images_dir in images_dirs:
            # it is possible the directory doesn't exist
            if not images_dir.exists():
                self._log.warning(f"Images directory {images_dir} doesn't exist")
                continue

            for image_file in images_dir.iterdir():
                if image_file.suffix not in [".png", ".jpg"]:
                    continue

                source_path = str(image_file)
                destination_path = str(person_dir / image_file.name)
                shutil.copy(source_path, destination_path)

                saved = True
            # end

        if saved:
            return person_id, person_dir
        else:
            self._log.warning(f"Images directories {images_dirs} doesn't exist")
            return -1, self.root / "invalid"
    # end

    def update_person_id(self, person_id, images_dirs):
        pass

    def _get_person_embeddings(self, person_id: int):
        if person_id not in self._people_embeddings:
            person_dir = self.root / str(person_id)
            person_embeddings = self._get_images_embedding([person_dir])
            self._people_embeddings[person_id] = person_embeddings
        return self._people_embeddings[person_id]

    def _get_images_embedding(self, images_dirs: list[Path]) -> list[EMBEDDING]:
        images_embedding: list[EMBEDDING] = []

        for images_dir in images_dirs:
            if not images_dir.exists():
                continue

            for image_file in images_dir.iterdir():
                if image_file.suffix not in [".png", ".jpg"]:
                    continue

                embedding: EMBEDDING = ClipReID.represent(image_file, self.model_name)
                images_embedding.append(embedding)
        # end
        return images_embedding


