import platform
import subprocess
import time
from typing import Tuple
import cv2
import numpy as np


class VideoStreamFFmpeg:
    def __init__(self, src: int, fps: int, resolution: Tuple[int, int]):
        self.src = src
        self.fps = fps
        self.resolution = resolution
        self.pipe = self._open_ffmpeg()
        self.frame_shape = (self.resolution[1], self.resolution[0], 3)
        self.frame_size = np.prod(self.frame_shape)
        self.wait_for_cam()

    def _open_ffmpeg(self):
        os_name = platform.system()
        if os_name == "Darwin":  # macOS
            input_format = "avfoundation"
            video_device = f"{self.src}:none"
        elif os_name == "Linux":
            input_format = "v4l2"
            video_device = f"{self.src}"
        elif os_name == "Windows":
            input_format = "dshow"
            video_device = f"video={self.src}"
        else:
            raise ValueError("Unsupported OS")

        command = [
            'ffmpeg',
            '-f', input_format,
            '-r', str(self.fps),
            '-video_size', f'{self.resolution[0]}x{self.resolution[1]}',
            '-i', video_device,
            '-vcodec', 'mjpeg',  # Input codec set to mjpeg
            '-an', '-vcodec', 'rawvideo',  # Decode the MJPEG stream to raw video
            '-pix_fmt', 'bgr24',
            '-vsync', '2',
            '-f', 'image2pipe', '-'
        ]
        print(" ".join(command))

        if os_name == "Linux":
            command.insert(2, "-input_format")
            command.insert(3, "mjpeg")

        return subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )

    def read(self):
        raw_image = self.pipe.stdout.read(self.frame_size)
        if len(raw_image) != self.frame_size:
            return None
        image = np.frombuffer(raw_image, dtype=np.uint8).reshape(self.frame_shape)
        return image

    def release(self):
        self.pipe.terminate()

    def wait_for_cam(self, max_count=1000):
        count = 0
        frame = None
        while count < max_count and frame is None:
            frame = self.read()
            count += 1
        return frame is not None
# end


class VideoStreamCV:
    def __init__(self, src: int, fps: int, resolution: Tuple[int, int]):
        self.src = src
        self.fps = fps
        self.resolution = resolution
        self.cap = self._open_camera()
        self.wait_for_cam()

    def _open_camera(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

    def wait_for_cam(self):
        for _ in range(30):
            frame = self.read()
        if frame is not None:
            return True
        return False


def main():
    fsp = 30
    resolution = (1920, 1080)

    cam = VideoStreamFFmpeg(src=0, fps=fsp, resolution=resolution)
    while True:
        frame = cam.read()
        pass

# end


if __name__ == "__main__":
    main()
