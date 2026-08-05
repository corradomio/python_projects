import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

import stdlib.loggingx as logging
from stdlib import jsonx
from stdlib.is_instance import is_instance
from stdlib.jsonx import JSONConfiguration
from stdlib.qname import create_from
from .utils import EMBEDDING, METRIC_TYPES, sort_tracks, LINKAGE_TYPE, FaceServer

TOP_SIMILARITY = 0.9999

# ---------------------------------------------------------------------------
# General idea:
#
#  The face solver scan a folder (in recursive way? Why not!) having the
#  following structure:
#
#       <faces-root>
#           <db-name>
#               <person-name>
#                   <face-image-1>.jpg
#                   ...
#                   person*.json     (optional)
#           ...
#
# (two levels is enough) to find the most similar image. When found, it uses the
# following person 'name':
#
#       "<db-name> <person-name>"
#
# IF "person*.json" is NOT available, otherwise the value of "FullName".
# The file "person*.json" is  JSON file starting with "person" BUT it can contain
# other characters. This is useful to simplify the manual assignment of a name
# to other folders.
# The structure of the file, for now, it is very simple:
#
#       {
#           "user": {
#               "FullName": "<full name>"
#           }
#       }
#
# This structure is compatible with the structure of Khalifa KU Face Recognition System
# returning a JSON file having the structure:
#
#   {
#       "labAccessData": [
#           {
#               "user": {
#                   "email": "100060593@ku.ac.ae",
#                   "personId": "100060593",
#                   "FullName": "Yazan  Hani Mousa  Abuhasheesh",
#                   "personType": "Student",
#                   "personCategory": "Doctorate",
#                   "personLevel": "PH",
#                   "jobDescription": "PhD in Engineering",
#                   "adAccount": "100060593",
#                   "gender": "M",
#                   "dateOfBirth": "1999-03-26",
#                   "nationality": "Jordan"
#               }
#               ...
#           },
#           ...
#       ]
#   }
#

#
# faces_store
#   faces_db
#       person_name
#   (faces_db, person_name): db_person

# Every time the module is used, it is created a new 'faces database' with
# name <db_name><index>
#
# If a face is recognized between the faces already registered in the 'faces databases'
# it is possible to update the list of faces using 'update_faces = true'
#
# If a face is not recognized, it is created a new entry in the last 'faces database'
# with name 'Person-<index>', containing the new faces
#
# Before to  terminate the analysis, the 'faces databases' are 'cleaned' from the
# duplicated images, based on 'same_similarity'. This permits to ensure a limited
# number of faces for each person
#

# ---------------------------------------------------------------------------
# FaceNameSolver
# ---------------------------------------------------------------------------

class FaceNameSolver(FaceServer):
    def __init__(self, CONFIG: JSONConfiguration, fsid: str=""):
        # fsid: useful in parallel processing
        super().__init__(CONFIG, "face_name_solver")

        self.enabled: bool = CONFIG.get("face_name_solver.enabled", False)
        self.faces_store: Path = Path(CONFIG.get("face_name_solver.faces_store", ".faces_store"))
        self.db_name = CONFIG.get("face_name_solver.db_name", "faces")

        # it can be used to reduce the length of the embedding
        self.embedding_length = CONFIG.get("face_name_solver.embedding_length", -1)

        self.embedding: dict = CONFIG.get("face_name_solver.embedding", {
            "class": "human.clipreid.ClipReID",
            "model_name": "DukeMTMC__cnn_base"
        })
        self.metric: str = CONFIG.get("face_name_solver.distance_metric", "cosine")

        # similarity between the images in the current track AND the images in the database
        self.similarity_threshold: float = CONFIG.get("face_name_solver.similarity_threshold", 0.97)

        # how to compare the images in two different folders:
        #   "average": using the average embeddings,
        #   "complete": farthest-neighbor or maximum distance method.
        #   "single": nearest-neighbor or minimum distance method.
        self.linkage: str = CONFIG.get("face_name_solver.linkage", "complete")

        # add the new faces to the folder containing the person identified
        self.update_faces: bool = CONFIG.get("face_name_solver.update_faces", False)

        # similarity to says: these two images are the same
        self.same_similarity: float = CONFIG.get("face_name_solver.same_similarity", 0.985)

        # 'same' subfolder contains images VERY similar ('same_similarity') to the images in the main folder
        # If necessary these images can be 're-analyzed'
        self.reanalyze_same: bool = CONFIG.get("face_name_solver.reanalyze_same", False)

        assert is_instance(self.metric, METRIC_TYPES)
        assert is_instance(self.linkage, LINKAGE_TYPE)
        assert is_instance(self.similarity_threshold, float)
        assert is_instance(self.same_similarity, float)
        assert self.same_similarity > self.similarity_threshold

        self.embedding_model = create_from(CONFIG.get("face_name_solver.embedding"))

        self._track_root: Path = None
        self._faces_embeddings: dict[Path, EMBEDDING] = {}
        self._faces_database: dict[tuple[str, str], list[EMBEDDING]] = {}
        self._means_database: dict[tuple[str, str], list[EMBEDDING]] = {}
        self._facesdb: str = ""

        self._log = logging.getLogger("FaceNameSolver"+fsid)

    def analyze(self, tracks_root: Path, db_name=""):
        if db_name is None or len(db_name) == 0:
            db_name = self.db_name

        assert is_instance(tracks_root, Path)
        assert is_instance(db_name, str)
        assert not db_name.startswith("/"), "dbname must not start with '/'"

        if not self.enabled:
            return

        self._log.info(f"Analyzing {tracks_root} ...")

        self._track_root = tracks_root
        self._faces_images: dict[tuple[str, str], list[Path]] = {}
        self._faces_database: dict[tuple[str, str], list[EMBEDDING]] = {}
        self._means_database: dict[tuple[str, str], list[EMBEDDING]] = {}

        # create the face_db (a directory)
        # self.faces_store = self.faces_store.parent / (self.faces_store.name + db_suffix)
        self.faces_store.mkdir(parents=True, exist_ok=True)

        # preload the faces database
        self._log.info("Loading faces database ...")
        self._load_faces_database()
        self._create_new_database(db_name)

        # scan the tracks
        track_dirs = [
            track_dir
            for track_dir in tracks_root.iterdir()
            if self._is_track_valid(track_dir)
        ]
        track_dirs: list[Path] = sort_tracks(track_dirs)
        n = len(track_dirs)

        self._log.infof(f"Analyzing tracks using linkage='{self.linkage}' ...")
        for i, track_dir in enumerate(track_dirs):
            face_dir = track_dir / "face"
            if not face_dir.is_dir(): continue

            self._log.infot(f"... {track_dir.name} ({i+1:4}/{n})")

            # faces_embeddings: list[EMBEDDING] = []
            # faces_files: list[Path] = []
            faces_embeddings, faces_files = self._get_faces_embedding(face_dir)

            if len(faces_embeddings) == 0:
                # self._log.warning(f"... ... no faces available for {track_dir.name}: skipped")
                continue

            db_person, similarity = self._find_person(faces_embeddings, faces_files)
            # db_person: tuple[str, str] = (db, person)
            # similarity: float

            if similarity <= self.similarity_threshold:
                db_person = self._create_new_person(face_dir, faces_embeddings, faces_files, db_person, similarity)
                self._log.info(f"... ... created new person: {db_person} (similarity={similarity:.3})")
            elif similarity >= self.same_similarity:
                self._log.infot(f"... ... already registered: {db_person}")
                pass
            elif self.update_faces:
                self._log.infot(f"... ... update person: {db_person} (similarity={similarity:.3})")
                self._update_person(db_person, face_dir, faces_embeddings)
            pass
        # end

        self._cleanup_faces_store()

        self._log.info(f"Done")
        self._cleanup()
    # end

    def _is_track_valid(self, track_dir: Path) -> bool:
        if not super()._is_track_valid(track_dir):
            return False

        # face directory must exist
        if not (track_dir / "face").exists():
            return False

        return True

    def _load_faces_database(self):
        # TWO levels
        for faces_db_dir in self.faces_store.iterdir():
            for person_dir in faces_db_dir.iterdir():
                self._log.infot(f"... ... {faces_db_dir.name} {person_dir.name }")
                face_embeddings, face_images = self._get_faces_embedding(person_dir)

                self._faces_database[(faces_db_dir.name, person_dir.name)] = face_embeddings
                self._means_database[(faces_db_dir.name, person_dir.name)] = [np.array(face_embeddings).mean(axis=0)]
            pass
        pass

    def _create_new_database(self, db_suffix: str):
        if len(db_suffix) == 0:
            db_suffix = self.db_name
        if len(db_suffix) == 0:
            db_suffix = "faces"

        linkage = self.linkage

        # start with '1'. In such way '0' can be used for 'service' reasons
        i = 1
        # facesdb = f"{db_suffix}-{linkage}-{i:02}"
        facesdb = f"{db_suffix}-{i:02}"
        while (self.faces_store / facesdb).exists():
            i += 1
            # facesdb = f"{db_suffix}-{linkage}-{i:02}"
            facesdb = f"{db_suffix}-{i:02}"

        self._facesdb = facesdb

    def _get_faces_embedding(self, person_dir: Path) -> tuple[list[EMBEDDING], list[Path]]:
        faces_embeddings: list[EMBEDDING] = []
        faces_files: list[Path] = []
        for face_file in person_dir.iterdir():
            # WARNING: the directory can contain subdirectories OR other files!
            if face_file.suffix != ".jpg": continue

            embedding = self._get_face_embedding(face_file)

            faces_embeddings.append(embedding)
            faces_files.append(face_file)
        return faces_embeddings, faces_files

    def _get_face_embedding(self, face_file:Path) -> EMBEDDING:

        if face_file in self._faces_embeddings:
            return self._faces_embeddings[face_file]

        embedding: EMBEDDING = self.embedding_model.embedding(face_file)

        # clip the embedding if longer than 'embedding_length'
        elen = self.embedding_length
        if 0 < elen < len(embedding):
            embedding = embedding[:elen]

        self._faces_embeddings[face_file] = embedding
        return embedding

    def _find_person(self, face_embeddings: list[EMBEDDING], faces_files: list[Path]) -> tuple[tuple[str, str], float]:
        assert len(face_embeddings) > 0

        best_name: tuple[str, str] = (self._facesdb, "unknown")
        best_similarity = 0.
        for db_person in self._faces_database.keys():
            if self.linkage == "complete":
                dist_matrix = cdist(face_embeddings, self._faces_database[db_person], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.max()
            elif self.linkage == "centroid":
                mean_faces_embeddings = [np.array(face_embeddings).mean(axis=0)]
                dist_matrix = cdist(mean_faces_embeddings, self._means_database[db_person], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.max()
            elif self.linkage == "single":
                dist_matrix = cdist(face_embeddings, self._faces_database[db_person], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.min()
            elif self.linkage == "average":
                dist_matrix = cdist(face_embeddings, self._faces_database[db_person], metric=self.metric)
                sim_matrix = 1 - dist_matrix
                similarity = sim_matrix.mean()
            else:
                raise ValueError(f"Unknown linkage: {self.linkage}")

            if similarity > best_similarity:
                best_name = db_person
                best_similarity = similarity
        # end
        return best_name, best_similarity

    def _create_new_person(
        self, face_dir: Path, faces_embeddings: list[EMBEDDING], faces_files: list[Path],
        best_db_person: tuple[str, str], best_similarity: float
    ) -> tuple[str, str]:
        last_person_id = self._find_last_person_id()
        person_name = f"Person-{last_person_id + 1}"

        person_dir = self.faces_store / self._facesdb / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(face_dir), str(person_dir), dirs_exist_ok=True)

        jsonx.dump({
            "person_name": list(best_db_person),
            "similarity": best_similarity,
            "threshold": self.similarity_threshold,
            "linkage": self.linkage,
            "metric": self.metric,
            "embedding": self.embedding,
        },
            (person_dir / "similarity.json"))

        # faces_embeddings = self._get_images_embedding(person_dir)
        means_embeddings = [np.array(faces_embeddings).mean(axis=0)]

        self._faces_database[(self._facesdb, person_name)] = faces_embeddings
        self._means_database[(self._facesdb, person_name)] = means_embeddings
        self._faces_images[(self._facesdb, person_name)] = faces_files

        return self._facesdb, person_name
    # end

    def _update_person(self, db_person: tuple[str, str], face_dir: Path, update_embeddings: list[EMBEDDING]):
        # if not self.update_faces: return

        face_db, person_name = db_person
        person_dir = self.faces_store / face_db / person_name
        shutil.copytree(str(face_dir), str(person_dir), dirs_exist_ok=True)

        # faces_embeddings = self._get_images_embedding(person_dir)
        faces_embeddings = self._faces_database[db_person] + update_embeddings
        means_embeddings = [np.array(faces_embeddings).mean(axis=0)]

        self._faces_database[db_person] = faces_embeddings
        self._means_database[db_person] = means_embeddings
        pass

    def _find_last_person_id(self) -> int:
        last_person_id = 0
        faces_dir = self.faces_store / self._facesdb
        faces_dir.mkdir(parents=True, exist_ok=True)

        for person_name in faces_dir.iterdir():
            if person_name.name.startswith("Person-"):
                person_id = int(person_name.name[len("Person-"):])
                if person_id > last_person_id:
                    last_person_id = person_id
        return last_person_id

    def _cleanup_faces_store(self):
        if self.same_similarity <= 0:
            return

        # faces_dir = self.faces_store / self._facesdb
        self._log.info("Cleanup faces_store ...")

        for faces_dir in self.faces_store.iterdir():
            self._log.info(f"... ... {faces_dir.name}")
            for person_dir in faces_dir.iterdir():
                if not person_dir.is_dir() or not person_dir.name.startswith("Person-"):
                    continue
                self._log.infot(f"... ... ... {person_dir.name}")
                self._cleanup_person_faces(person_dir)
        pass

    def _cleanup_person_faces(self, person_dir: Path):

        same_dir = person_dir / "same"
        same_threshold = 1 - self.same_similarity
        embedding_map: dict[str, EMBEDDING] = {}

        # 0) WE SUPPOSE that IF the analysis is applied another time one the SAME folder
        #    containing 'same' subdirectory', THEN, the images in 'same' MUST BE REANALIZED
        if same_dir.exists() and self.reanalyze_same:
            for image_file in same_dir.iterdir():
                shutil.move(image_file, person_dir)

        # 1) load embeddings
        for face_file in person_dir.iterdir():
            if face_file.suffix != ".jpg": continue

            embedding: EMBEDDING = self._get_face_embedding(face_file)
            embedding_map[face_file.name] = embedding
        # end
        n = len(embedding_map)

        # 2) compose embedding list
        file_names = list(embedding_map.keys())
        embeddings = [embedding_map[file_name] for file_name in file_names]

        # 3) compute distance map
        dist_matrix = cdist(embeddings, embeddings, metric="cosine")

        # 4) identifies too similar  images
        too_similar: list[tuple[int, int, float]] = []
        excluded: set[int] = set()
        for i in range(n):
            if i in excluded: continue
            for j in range(i + 1, n):
                if j in excluded: continue

                dij = dist_matrix[i, j]
                if dij <= same_threshold:
                    too_similar.append((i, j, dij))
                    excluded.add(j)
        # end

        if len(too_similar) == 0:
            return

        # 5) print logs
        self._log.info(f"... too_similar: {len(too_similar)}")
        for i, j, dij in too_similar:
            self._log.debugt(f"... ... {file_names[i]} - {file_names[j]}: {dij}")

        # 6) MOVE the images to ignore
        same_dir.mkdir(parents=True, exist_ok=True)
        same_map: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for i, j, dij in too_similar:
            from_name = file_names[i]
            move_name = file_names[j]
            move_file = person_dir / move_name
            if not move_file.exists(): continue

            try:
                shutil.move(move_file, same_dir, )
                same_map[from_name].append((move_name, dij))
            except:
                pass
        # end

        # 7) save the statistics
        jsonx.dump(same_map, person_dir / "same.json")
    # end

    def _cleanup(self):
        # cleanup of the internal data structures
        self._faces_database: dict[str, list[EMBEDDING]] = {}
        self._means_database: dict[str, list[EMBEDDING]] = {}
        self._faces_embeddings: dict[Path, EMBEDDING] = {}
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
