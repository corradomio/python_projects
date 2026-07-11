from pathlib import Path
from typing import Optional

from human.pose_estimation import HumanPose, PoseKeypoints
from post_processing.utils import LabMonitoring, sort_tracks
from stdlib import jsonx
from stdlib import loggingx as logging
from stdlib.jsonx import JSONConfiguration


# ---------------------------------------------------------------------------
# PoseClassifier
# ---------------------------------------------------------------------------
# nose: naso
# eye: occhio
# ear: orecchio
# shouder: spalla
# elbow: gomito
# wrist: polso
# hip: anca
# knee: ginocchio
# ankle: caviglia
#                           x                  y                  score
# {
#         "nose":         [ 93.27582550048828, 66.10623931884766, 0.9682109951972961 ],
#         "left": {
#             "eye":      [108.21855163574219, 47.33680725097656, 0.9859724044799805 ],
#             "ear":      [ 152.58743286132812, 53.642391204833984, 0.988345742225647 ],
#             "shoulder": [ 199.6837615966797, 156.56724548339844, 0.9989621639251709 ],
#             "elbow":    [ 250.22711181640625, 319.5222473144531, 0.998615026473999 ],
#             "wrist":    [ 220.28875732421875, 450.9058532714844, 0.9961207509040833 ],
#             "hip":      [ 183.538818359375, 409.70013427734375, 0.9995840191841125 ],
#             "knee":     [ 193.41529846191406, 562.6109619140625, 0.9964165687561035 ],
#             "ankle":    [ 298.4483337402344, 673.1411743164062, 0.9066886305809021 ]
#         },
#         "right": {
#             "eye":      [ 86.8448486328125, 51.8585090637207, 0.4282367527484894 ],
#             "ear":      [ 96.38050842285156, 60.55195617675781, 0.027955587953329086 ],
#             "shoulder": [ 108.03970336914062, 157.57217407226562, 0.947382390499115 ],
#             "elbow":    [ 122.10420227050781, 300.04559326171875, 0.7101078629493713 ],
#             "wrist":    [ 116.35198974609375, 392.32080078125, 0.7728708386421204 ],
#             "hip":      [ 111.00658416748047, 405.4611511230469, 0.9966709017753601 ],
#             "knee":     [ 58.690330505371094, 562.1575317382812, 0.974157989025116 ],
#             "ankle":    [ 96.72418975830078, 672.6315307617188, 0.7397366762161255 ]
#         }
#     }

class PoseClassifier:

    @staticmethod
    def load(pose_file: Path, threshold: float = 0.66):
        pose = jsonx.load(pose_file)
        return PoseClassifier(pose, threshold)

    def __init__(self, pose: dict, threshold: float = 0.66):
        self._pose = {} if pose is None else pose
        self._threshold = threshold

    def is_valid(self):
        pose =self._pose
        if len(pose) == 0:
            return False
        # at minimum a keypoint must be greater than threshold
        for k in pose:
            score = pose[k][2]
            if score > self._threshold:
                return True
        return False

    def which_pose(self) -> set[str]:
        return "any"
    
    def keypoints(self) -> set[str]:
        kp: set[str] = set()
        if len(self._pose) == 0:
            return kp
        if self._pose["nose"][2] >= self._threshold:
            kp.add("nose")
        if self._pose["left_eye"][2] >= self._threshold:
            kp.add("left_eye")
            kp.add("eye")
        if self._pose["right_eye"][2] >= self._threshold:
            kp.add("right_eye")
            kp.add("eye")
        if self._pose["left_ear"][2] >= self._threshold:
            kp.add("left_ear")
            kp.add("ear")
        if self._pose["right_ear"][2] >= self._threshold:
            kp.add("right_ear")
            kp.add("ear")
        if self._pose["left_shoulder"][2] >= self._threshold:
            kp.add("left_shoulder")
            kp.add("shoulder")
        if self._pose["right_shoulder"][2] >= self._threshold:
            kp.add("right_shoulder")
            kp.add("shoulder")
        if self._pose["left_elbow"][2] >= self._threshold:
            kp.add("left_elbow")
            kp.add("elbow")
        if self._pose["right_elbow"][2] >= self._threshold:
            kp.add("right_elbow")
            kp.add("elbow")
        if self._pose["left_wrist"][2] >= self._threshold:
            kp.add("left_wrist")
            kp.add("wrist")
        if self._pose["right_wrist"][2] >= self._threshold:
            kp.add("right_wrist")
            kp.add("wrist")
        if self._pose["left_hip"][2] >= self._threshold:
            kp.add("left_hip")
            kp.add("hip")
        if self._pose["right_hip"][2] >= self._threshold:
            kp.add("right_hip")
            kp.add("hip")
        if self._pose["left_knee"][2] >= self._threshold:
            kp.add("left_knee")
            kp.add("knee")
        if self._pose["right_knee"][2] >= self._threshold:
            kp.add("right_knee")
            kp.add("knee")
        if self._pose["left_ankle"][2] >= self._threshold:
            kp.add("left_ankle")
            kp.add("ankle")
        if self._pose["right_ankle"][2] >= self._threshold:
            kp.add("right_ankle")
            kp.add("ankle")
        return kp

    def pose_class(self) -> str:
        kp = self.keypoints()
        if len(kp) == 0:
            return "unknown"

# end


# ---------------------------------------------------------------------------
# PoseAnalyzer
# ---------------------------------------------------------------------------

class PoseAnalyzer(LabMonitoring):

    def __init__(self, CONFIG: JSONConfiguration):
        super().__init__(CONFIG, "pose_analyzer")

        self.enabled = CONFIG.get("pose_analyzer.enabled", False)
        self.image_folders = CONFIG.get("pose_analyzer.image_folders", ["random_crop"])
        self.model_name = CONFIG.get("pose_analyzer.model_name", "")
        self.save_pose = CONFIG.get("pose_analyzer.save_pose", True)

        self._tracks_root: Path = None
        self._image_pose_map: dict[Path, PoseKeypoints] = {}
        self._log = logging.getLogger("PoseAnalyzer")

    # -----------------------------------------------------------------------
    # analyze
    # -----------------------------------------------------------------------

    def analyze(self, tracks_root: Path):
        assert isinstance(tracks_root, Path)

        self._tracks_root = tracks_root
        self._image_pose_map = {}

        if not self.enabled: return

        self._log.info(f"Analyzing {tracks_root} ...")

        # 1) collect the list of track names to process
        track_dirs = [
            track_dir
            for track_dir in tracks_root.iterdir()
            if self._is_track_valid(track_dir)
        ]
        # 2) sort track names lexicographically
        track_dirs: list[Path] = sort_tracks(track_dirs)
        n = len(track_dirs)

        for i, track_dir in enumerate(track_dirs):
            # skip if 'pose.json' is already present
            if (track_dir / "pose.json").exists():
                continue

            self._log.infot(f"... {track_dir.name} ({i+1:4}/{n})")

            track_pose = self._analyze_track(track_dir)
            self._save(track_dir, track_pose)

        self._cleanup()
        self._log.info(f"Done")
    # end

    def _is_track_valid(self, track_dir: Path) -> bool:
        # if the track name is '<camid>_<trackid>_DONE'
        if not track_dir.name.endswith("_DONE"):
            return False

        # random_crop must exist
        if not (track_dir / "random_crop").exists():
            return False

        return True

    def _analyze_track(self, track_dir: Path) -> dict[str, PoseKeypoints]:
        # self._log.infot(f"... {track_dir.name}")

        track_pose: dict[str, PoseKeypoints] = {}

        for image_folder in self.image_folders:
            image_dir = track_dir / image_folder

            for image_path in image_dir.glob("*.jpg"):
                try:
                    image_pose: PoseKeypoints = HumanPose.pose(image_path, self.model_name)
                except Exception as e:
                    self._log.exception(f"Unable to infer the pose for {image_path}")
                    image_pose = PoseKeypoints(None)

                self._image_pose_map[image_path] = image_pose
                track_pose[image_path.name] = image_pose

        return track_pose
    # end

    def _save(self, track_dir: Path, track_pose: dict[str, PoseKeypoints]):
        if not self.save_pose: return

        data = {
            image_name: track_pose[image_name].as_dict()
            for image_name in track_pose
        }

        jsonx.dump(data, track_dir / "pose.json")
    # end

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    def is_image_valid(self, image_path: Path) -> bool:

        if not self.enabled: return True

        # an image is valid IF it contains valid keypoints
        # Note:we have to DECIDE when a pose is 'valid', that is
        # which keypoints are visible
        assert isinstance(image_path, Path)
        assert image_path in self._image_pose_map

        return self._image_pose_map[image_path].is_valid()

    def is_track_valid(self, track_dir: Path) -> bool:

        if not self.enabled: return True

        # a track is valid IF at minimum 1 image is valid
        assert isinstance(track_dir, Path)

        for image_folder in self.image_folders:
            image_dir = track_dir / image_folder

            for image_path in image_dir.glob("*.jpg"):
                if self.is_image_valid(image_path):
                    return True
        # end
        return False

    # -----------------------------------------------------------------------
    # get_image_pose
    # -----------------------------------------------------------------------

    def get_image_pose(self, image_path: Path) -> Optional[PoseKeypoints]:

        if not self.enabled: return None

        assert isinstance(image_path, Path)
        assert image_path in self._image_pose_map

        return self._image_pose_map[image_path]

    def _cleanup(self):
        HumanPose.dispose()
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

