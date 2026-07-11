from pathlib import Path
from typing import Optional, cast

import cv2
import face_detection
import numpy as np
from face_detection.torch_utils import get_device

from stdlib import loggingx as logging
from stdlib.is_instance import is_instance
from stdlib.jsonx import JSONConfiguration
from .utils import chop, IMAGE_ARRAY, LabMonitoring, sort_tracks

LINE_COLOR = (255, 0, 255)
LINE_THICKNESS = 2


# ---------------------------------------------------------------------------
# FaceDetection
# ---------------------------------------------------------------------------

class FaceDetection:
    def __init__(
            self,
            name: str = "DSFDDetector",
            confidence_threshold: float = 0.5,
            nms_iou_threshold: float = 0.3,
            device=get_device(),
            max_resolution: int = None,
            fp16_inference: bool = False,
            clip_boxes: bool = False
    ):
        self._model = face_detection.build_detector(
            name=name,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            max_resolution=max_resolution,
            fp16_inference=fp16_inference,
            clip_boxes=clip_boxes
        )

    def detect(self, image, shrink=1.0) -> np.ndarray:
        return self._model.detect(image, shrink=shrink)

    def plot(self, frame, boxes):
        def _box(box):
            tx, ty, bx, by, prob = box
            top_left_corner = (int(tx), int(ty))
            bottom_right_corner = (int(bx), int(by))
            cv2.rectangle(frame, top_left_corner, bottom_right_corner, LINE_COLOR, LINE_THICKNESS)

        for box in boxes:
            _box(box)
    # end
# end


# ---------------------------------------------------------------------------
# FaceExtractor
# ---------------------------------------------------------------------------
#
# FACE_SUBDIRS = ["face_recognition", "random_crop",
#                 "not_dress_well", "not_glove_well",
#                 # "unauthorised_access", "unauthorised_machine_touching_B"
#                 # "cleaner_or_security"
#                 ]

class FaceAnalyzer(LabMonitoring):

    def __init__(self, CONFIG: JSONConfiguration):
        super().__init__(CONFIG, "face_analyzer")

        self.enabled = CONFIG.get("face_analyzer.enabled", default=False)
        self.confidence_threshold = CONFIG.get("face_analyzer.confidence_threshold")
        self.nms_iou_threshold = CONFIG.get("face_analyzer.nms_iou_threshold")

        self.aspect_ratio = CONFIG.get("face_analyzer.aspect_ratio", 0.6)
        self.face_border = CONFIG.get("face_analyzer.face_border", 10)
        self.face_pixels = CONFIG.get("face_analyzer.face_pixels", 40*40)
        self.image_folders = CONFIG.get("face_analyzer.image_folders")
        self.excluding = CONFIG.get("face_analyzer.excluding")
        self.flip_face = CONFIG.get("face_analyzer.flip_face", False)

        self.face_detection = FaceDetection(
            "DSFDDetector",
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold
        )

        self._log = logging.getLogger("FaceAnalyzer")
        pass

    def analyze(self, tracks_root: Path, face_db_suffix=""):
        # Analyze the images in
        #   <root_path>/<folder>/face_recognition
        #   <root_path>/<folder>/random_crop
        #
        # where <folder> has the form '<camid>_<trackid>_DONE'
        # The faces are saved in
        #   <root_path>/<folder>/faces
        #
        # Note, the image name has the structure:
        assert is_instance(tracks_root, Path)

        if not self.enabled: return

        log = self._log

        log.info(f"Analyzing {tracks_root} ...")

        track_dirs = [
            track_dir
            for track_dir in tracks_root.iterdir()
            if self._is_track_valid(track_dir)
        ]
        track_dirs: list[Path] = sort_tracks(track_dirs)
        n = len(track_dirs)

        for i, track_dir in enumerate(track_dirs):
            log.infot(f"... {track_dir.name} ({i+1:4}/{n})")

            if (track_dir / "no_face").exists() or (track_dir / "face").exists():
                continue

            (track_dir / "no_face").touch(exist_ok=True)

            # WARNING: NO!!!
            # to create the directory ONLY if some face is saved inside!
            # face_dir.mkdir(parents=True, exist_ok=True)

            for images_name in self.image_folders:
                images_folder = track_dir / images_name
                if not images_folder.exists():
                    continue

                for image_file in images_folder.glob("*.jpg"):
                    if self._exists_face(image_file):
                        continue
                    if self._excluded_image(image_file):
                        continue

                    face = self._extract_face(image_file)
                    if face is None:
                        continue

                    self._save_face(image_file, face)
                # end
            # end
        # end
        self._cleanup()
        log.info(f"Done")
    # end

    def _is_track_valid(self, track_dir: Path) -> bool:
        # if the track name is '<camid>_<trackid>_DONE'
        if not track_dir.name.endswith("_DONE"):
            return False

        # random_crop must exist
        if not (track_dir / "random_crop").exists():
            return False

        return True

    def _excluded_image(self, image_file: Path):
        image_name = image_file.name
        for pat in self.excluding:
            if pat in image_name:
                return True
        return False

    def _exists_face(self, image_file: Path) -> bool:
        folder_path = image_file.parent.parent
        face_path = folder_path / f"face/face_{image_file.name}"
        return face_path.exists()

    def _extract_face(self, image_path: Path) -> Optional[IMAGE_ARRAY]:
        log = self._log

        face_detection = self.face_detection
        aspect_ratio = self.aspect_ratio
        face_border = self.face_border
        face_pixels = self.face_pixels

        image: IMAGE_ARRAY = cast(IMAGE_ARRAY, cv2.imread(str(image_path)))

        boxes = face_detection.detect(image)
        if len(boxes) == 0:
            return None

        if len(boxes) > 1:
            pass

        # boxes: (1, 5) -> x1,y1,x2,y2, probability
        x1,y1,x2,y2, prob = boxes[0]
        h, w, c = image.shape

        dx = (w*face_border) if face_border < 1 else face_border
        dy = (h*face_border) if face_border < 1 else face_border

        x1 = chop(x1-dx, w)
        y1 = chop(y1-dy, h)
        x2 = chop(x2+dx, w)
        y2 = chop(y2+dy, h)

        w = x2-x1
        h = y2-y1
        if (w*h) < face_pixels or (w/h) < aspect_ratio or (h/w) < aspect_ratio:
            return None

        log.infot(f"... ... {image_path.name}")

        face = image[y1:y2, x1:x2]

        if x1==x2 or y1==y2:
            return None

        return face

    def _save_face(self, image_file: Path, face: IMAGE_ARRAY):
        try:
            assert is_instance(image_file, Path)

            # image_file: <track_dir>/<image_folder>/<image_file>.jpg
            track_dir: Path = image_file.parent.parent

            (track_dir / "no_face").unlink(missing_ok=True)

            face_root = track_dir / "face"
            face_root.mkdir(exist_ok=True)

            face_path = face_root / f"face_{image_file.name}"
            if not face_path.exists():
                cv2.imwrite(str(face_path), face)

            if not self.flip_face:
                return

            flip_path = face_root / f"flip_{image_file.name}"
            flip_face = cv2.flip(face, 1)  # flip horizontally
            if not flip_path.exists():
                cv2.imwrite(str(flip_path), flip_face)
            pass
        except Exception as e:
            self._log.exception(f"Unable to save {image_file}: {face.shape}")

    def _cleanup(self):
        pass
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
