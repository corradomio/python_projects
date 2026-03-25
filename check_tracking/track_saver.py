import numpy as np
import cv2

XYXY_ARRAY = np.ndarray


class TrackSaver:
    def __init__(self, save_dir: str, camera_id: int=0):
        self._save_dir = save_dir
        self._camera_id = camera_id

    def save(self, frame: np.ndarray, tracks: dict[int, XYXY_ARRAY], faces: np.ndarray, directions: np.ndarray):
        """

        :param frame: [h,w,c] image
        :param tracks: {id: [tx,ty,bx,by], ...}
        :param faces: [:,[tx,ty,bx,by,prob]]
        :param directions: [:, [pitch, yaw, roll]]
        :return:
        """
        timestamp =

        pass

