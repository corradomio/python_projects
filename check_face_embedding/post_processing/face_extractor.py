import re
from pathlib import Path
from typing import Optional, cast

import cv2
import face_detection
import numpy as np
from face_detection.torch_utils import get_device

from stdlib import loggingx as logging
from stdlib.is_instance import is_instance
from stdlib.jsonx import JSONConfiguration
from .utils import chop, IMAGE_ARRAY

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


class FaceExtractor:

    def __init__(self, CONFIG: JSONConfiguration):
        self.CONFIG = CONFIG

        self.enabled = CONFIG.get("face_extractor.enabled", default=False)
        self.override = CONFIG.get("face_extractor.override", default=False)
        self.confidence_threshold = CONFIG.get("face_extractor.confidence_threshold")
        self.nms_iou_threshold = CONFIG.get("face_extractor.nms_iou_threshold")

        self.aspect_ratio = CONFIG.get("face_extractor.aspect_ratio", 0.6)
        self.face_border = CONFIG.get("face_extractor.face_border", 10)
        self.face_pixels = CONFIG.get("face_extractor.face_pixels", 40*40)
        self.include_chest = CONFIG.get("face_extractor.include_chest", 0)
        self.image_folders = CONFIG.get("face_extractor.image_folders")
        self.excluding = CONFIG.get("face_extractor.excluding")
        self.flip_face = CONFIG.get("face_extractor.flip_face", False)

        self.face_detection = FaceDetection(
            "DSFDDetector",
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold
        )

        self.log = logging.getLogger("FaceExtractor")
        pass

    def extract_faces(self, root_path: str|Path, cam_id: int=-1):
        # Analyze the images in
        #   <root_path>/<folder>/face_recognition
        #   <root_path>/<folder>/random_crop
        #
        # where <folder> has the form '<cam_id>_<track_id>_DONE'
        # The faces are saved in
        #   <root_path>/<folder>/faces
        #
        # Note, the image name has the structure:
        assert is_instance(root_path, (str, Path))
        assert is_instance(cam_id, int)

        if not self.enabled:
            return

        log = self.log
        root_dir: Path = Path(root_path) if isinstance(root_path, str) else root_path
        subfolders = self.image_folders

        log.info(f"Starting face extraction in {root_dir}")

        pattern = f"{cam_id}_[^_]+_DONE" if cam_id >= 0 else "[^_]+_[^_]+_DONE"
        #for folder in root_dir.glob(pattern):
        for folder, _, _ in root_dir.walk():
            if not re.match(pattern, folder.name):
                continue

            log.infot(f"... {folder.name}")

            # face_dir = folder / "face"
            # if face_dir.exists():
            #     continue
            # else:
            #     face_dir.mkdir(exist_ok=False)

            for subdir in subfolders:
                images_folder = folder / subdir
                if not images_folder.exists():
                    continue

                for image_file in images_folder.glob("*.jpg"):
                    # if self._exists_face(image_file) and not self.override:
                    #     continue
                    if self._excluded_image(image_file):
                        continue

                    face = self._extract_face(image_file)
                    if face is None:
                        continue

                    try:
                        self._save_face(image_file, face)
                    except Exception as e:
                        log.exception(f"Unable to save {image_file}: {face.shape}")
                # end
            # end
        # end
        # log.info(f"Done")
    # end

    def _excluded_image(self, image_file: Path):
        image_name = image_file.name
        for pat in self.excluding:
            if pat in image_name:
                return True
        return False

    def _exists_face(self, image_file: Path) -> bool:
        folder_path = image_file.parent.parent
        face_path = folder_path / "face" / image_file.name
        return face_path.exists()

    def _extract_face(self, image_path: Path) -> Optional[IMAGE_ARRAY]:
        log = self.log

        face_detection = self.face_detection
        aspect_ratio = self.aspect_ratio
        face_border = self.face_border
        face_pixels = self.face_pixels
        include_chest = self.include_chest

        image: IMAGE_ARRAY = cast(IMAGE_ARRAY, cv2.imread(image_path))

        boxes = face_detection.detect(image)
        if len(boxes) == 0:
            return None

        if len(boxes) > 1:
            pass

        # boxes: (1, 5) -> x1,y1,x2,y2, probability
        x1,y1,x2,y2, prob = boxes[0]
        h, w, c = image.shape

        if include_chest > 0:
            dy = y2-y1
            y2 = y1 + include_chest*dy

        dx = (w*face_border) if face_border < 1 else face_border
        dy = (h*face_border) if face_border < 1 else face_border

        x1 = chop(x1-dx, w)
        y1 = chop(y1-dy, h)
        x2 = chop(x2+dx, w)
        y2 = chop(y2+dy, h)

        w = x2-x1
        h = y2-y1
        if w == 0 or h == 0 or (w*h) < face_pixels or (w/h) < aspect_ratio or (h/w) < aspect_ratio:
            return None

        log.infot(f"... ... {image_path.name}")

        face = image[y1:y2, x1:x2]

        return face
    # end

    def _save_face(self, image_path: Path, face: IMAGE_ARRAY):
        assert is_instance(image_path, Path)

        # image_path: ...<folder>/face_recognition/<image>.jpg
        folder_root = image_path.parent.parent

        face_root = folder_root / "face"
        face_root.mkdir(parents=True, exist_ok=True)
        face_path = face_root / image_path.name

        if not face_path.exists() or self.override:
            cv2.imwrite(face_path, face)

        if not self.flip_face:
            return

        flip_path = face_root / f"{image_path.stem}_flip{image_path.suffix}"
        flip_face = cv2.flip(face, 1)  # horizontally
        if not flip_path.exists() or self.override:
            cv2.imwrite(flip_path, flip_face)
        pass
    # end
# end
