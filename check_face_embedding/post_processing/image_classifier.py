import shutil
from pathlib import Path

import stdlib.loggingx as logging
from .image_embedding import ImageEmbedding
from .lab_dress_glove import classify_image
from .utils import LabMonitoring, sort_tracks


# ---------------------------------------------------------------------------
# ImageClassifier
# ---------------------------------------------------------------------------

class ImageClassifier(LabMonitoring):

    def __init__(
        self,
        CONFIG,
        image_embedding: ImageEmbedding
    ):
        super().__init__(CONFIG, "image_classifier")

        assert isinstance(image_embedding, ImageEmbedding)

        self.image_embedding: ImageEmbedding = image_embedding

        self.enabled: bool = CONFIG.get("image_classifier.enabled", False)

        self.image_store = Path(CONFIG.get("image_classifier.image_store", ".images_classification"))
        
        self.glove_threshold = CONFIG.get("image_classifier.glove_threshold", 0.25)
        self.dress_threshold = CONFIG.get("image_classifier.dress_threshold", 0.85)
        self.cleaner_security_threshold = CONFIG.get("image_classifier.cleaner_security_threshold", 0.92)

        self.max_similarity = CONFIG.get("image_classifier.max_similarity", 0.98)
        self.max_images = CONFIG.get("image_classifier.max_images", 3)

        self._track_root: Path = None
        self._log = logging.getLogger("ImageClassifier")

    def analyze(self, tracks_root: Path, db_suffix=""):
        assert isinstance(tracks_root, Path)
        assert isinstance(db_suffix, str)

        if not self.enabled: return

        self._log.info(f"Analyzing {tracks_root} ...")

        self._track_root = tracks_root

        # create the image store (a directory
        self.image_store = self.image_store.parent / (self.image_store.name + db_suffix)

        self._create_classification_dirs()

        # 1) collect the list of track names to process
        track_dirs = [
            track_dir
            for track_dir in tracks_root.iterdir()
            if self._is_track_valid(track_dir)
        ]
        track_dirs: list[Path] = sort_tracks(track_dirs)
        n = len(track_dirs)

        for i, track_dir in enumerate(track_dirs):
            self._log.infot(f"... {track_dir.name} ({i+1:4}/{n})")

            selected_images: list[Path] = self.image_embedding.get_track_selected_images(track_dir)
            selected_images = self._filter_images_on_similarity(selected_images)

            for image_path in selected_images:
                self._classify_image(image_path)
        # end
        self._log.info(f"Done")
    pass

    def _is_track_valid(self, track_dir: Path):
        # if the track name is '<camid>_<trackid>_DONE'
        if not track_dir.name.endswith("_DONE"):
            return False

        if len(self.image_embedding.get_track_selected_images(track_dir)) == 0:
            return False

        return True
    # end

    def _create_classification_dirs(self):
        image_store = self.image_store
        image_store.mkdir(parents=True, exist_ok=True)

        (image_store / "person").mkdir(parents=True, exist_ok=True)
        (image_store / "cleaner_security").mkdir(parents=True, exist_ok=True)

        (image_store / "not_well_dress").mkdir(parents=True, exist_ok=True)
        (image_store / "well_dress").mkdir(parents=True, exist_ok=True)

        (image_store / "not_glove_well").mkdir(parents=True, exist_ok=True)
        (image_store / "glove_well").mkdir(parents=True, exist_ok=True)
    # end

    def _filter_images_on_similarity(self, selected_images: list[Path]) -> list[Path]:
        if len(selected_images) > self.max_images:
            selected_images = self.image_embedding.filter_same_images(selected_images, self.max_similarity)
        return selected_images
    # end

    def _classify_image(self, image_path: Path):

        classifications = classify_image(image_path)
        # [[person_type], [dress_type], [glove_type]]
        if classifications is None:
            return

        # classes:
        #   glove:  1,3,7   > conf_glove_thr=0.25
        #   dress   0, 2    > conf_dress_thr=0.85
        #   person: 4, 5, 6 > conf_cleaner_security_thr=0.92

        # converted in:
        #   [person, cleaner_security]
        #   [not_well_dress, well_dress]
        #   [not_glove_well, glove_well]

        person, cleaner_security = classifications[0]
        not_well_dress, well_dress = classifications[1]
        not_glove_well, glove_well = classifications[1]

        if cleaner_security >= self.cleaner_security_threshold:
            self._save_in(image_path, "cleaner_security")
        else:
            self._save_in(image_path, "person")

        if well_dress >= self.dress_threshold:
            self._save_in(image_path, "well_dress")
        else:
            self._save_in(image_path, "not_well_dress")

        if glove_well >= self.glove_threshold:
            self._save_in(image_path, "glove_well")
        else:
            self._save_in(image_path, "not_glove_well")

    def _save_in(self, image_path, classification: str):
        save_folder = self.image_store / classification
        shutil.copy(str(image_path), str(save_folder))
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
