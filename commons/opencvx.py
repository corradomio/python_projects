import time
from multiprocessing import Process, Value, Queue, SimpleQueue
from threading import Thread
from typing import Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

#
# OpenCV extensions and utilities
#
VIDEO_CAPTURE_APIS = {
    cv2.CAP_ANY: "CAP_ANY",
    cv2.CAP_VFW: "CAP_VFW",
    cv2.CAP_V4L: "CAP_V4L",
    cv2.CAP_V4L2: "CAP_V4L2",
    cv2.CAP_FIREWIRE: "CAP_FIREWIRE",
    cv2.CAP_FIREWARE: "CAP_FIREWARE",
    cv2.CAP_IEEE1394: "CAP_IEEE1394",
    cv2.CAP_DC1394: "CAP_DC1394",
    cv2.CAP_CMU1394: "CAP_CMU1394",
    cv2.CAP_QT: "CAP_QT",
    cv2.CAP_UNICAP: "CAP_UNICAP",
    cv2.CAP_DSHOW: "CAP_DSHOW",
    cv2.CAP_PVAPI: "CAP_PVAPI",
    cv2.CAP_OPENNI: "CAP_OPENNI",
    cv2.CAP_OPENNI_ASUS: "CAP_OPENNI_ASUS",
    cv2.CAP_ANDROID: "CAP_ANDROID",
    cv2.CAP_XIAPI: "CAP_XIAPI",
    cv2.CAP_AVFOUNDATION: "CAP_AVFOUNDATION",
    cv2.CAP_GIGANETIX: "CAP_GIGANETIX",
    cv2.CAP_MSMF: "CAP_MSMF",
    cv2.CAP_WINRT: "CAP_WINRT",
    cv2.CAP_INTELPERC: "CAP_INTELPERC",
    cv2.CAP_REALSENSE: "CAP_REALSENSE",
    cv2.CAP_OPENNI2: "CAP_OPENNI2",
    cv2.CAP_OPENNI2_ASUS: "CAP_OPENNI2_ASUS",
    cv2.CAP_OPENNI2_ASTRA: "CAP_OPENNI2_ASTRA",
    cv2.CAP_GPHOTO2: "CAP_GPHOTO2",
    cv2.CAP_GSTREAMER: "CAP_GSTREAMER",
    cv2.CAP_FFMPEG: "CAP_FFMPEG",
    cv2.CAP_IMAGES: "CAP_IMAGES",
    cv2.CAP_ARAVIS: "CAP_ARAVIS",
    cv2.CAP_OPENCV_MJPEG: "CAP_OPENCV_MJPEG",
    cv2.CAP_INTEL_MFX: "CAP_INTEL_MFX",
    cv2.CAP_XINE: "CAP_XINE",
    cv2.CAP_UEYE: "CAP_UEYE",
    cv2.CAP_OBSENSOR: "CAP_OBSENSOR",
}

VIDEO_CAPTURE_PROPERTIES: dict[int, str] = {
    cv2.CAP_PROP_POS_MSEC: "CAP_PROP_POS_MSEC",
    cv2.CAP_PROP_POS_FRAMES: "CAP_PROP_POS_FRAMES",
    cv2.CAP_PROP_POS_AVI_RATIO: "CAP_PROP_POS_AVI_RATIO",
    cv2.CAP_PROP_FRAME_WIDTH: "CAP_PROP_FRAME_WIDTH",
    cv2.CAP_PROP_FRAME_HEIGHT: "CAP_PROP_FRAME_HEIGHT",
    cv2.CAP_PROP_FPS: "CAP_PROP_FPS",
    cv2.CAP_PROP_FOURCC: "CAP_PROP_FOURCC",
    cv2.CAP_PROP_FRAME_COUNT: "CAP_PROP_FRAME_COUNT",
    cv2.CAP_PROP_FORMAT: "CAP_PROP_FORMAT",
    cv2.CAP_PROP_MODE: "CAP_PROP_MODE",
    cv2.CAP_PROP_BRIGHTNESS: "CAP_PROP_BRIGHTNESS",
    cv2.CAP_PROP_CONTRAST: "CAP_PROP_CONTRAST",
    cv2.CAP_PROP_SATURATION: "CAP_PROP_SATURATION",
    cv2.CAP_PROP_HUE: "CAP_PROP_HUE",
    cv2.CAP_PROP_GAIN: "CAP_PROP_GAIN",
    cv2.CAP_PROP_EXPOSURE: "CAP_PROP_EXPOSURE",
    cv2.CAP_PROP_CONVERT_RGB: "CAP_PROP_CONVERT_RGB",
    cv2.CAP_PROP_WHITE_BALANCE_BLUE_U: "CAP_PROP_WHITE_BALANCE_BLUE_U",
    cv2.CAP_PROP_RECTIFICATION: "CAP_PROP_RECTIFICATION",
    cv2.CAP_PROP_MONOCHROME: "CAP_PROP_MONOCHROME",
    cv2.CAP_PROP_SHARPNESS: "CAP_PROP_SHARPNESS",
    cv2.CAP_PROP_AUTO_EXPOSURE: "CAP_PROP_AUTO_EXPOSURE",
    cv2.CAP_PROP_GAMMA: "CAP_PROP_GAMMA",
    cv2.CAP_PROP_TEMPERATURE: "CAP_PROP_TEMPERATURE",
    cv2.CAP_PROP_TRIGGER: "CAP_PROP_TRIGGER",
    cv2.CAP_PROP_TRIGGER_DELAY: "CAP_PROP_TRIGGER_DELAY",
    cv2.CAP_PROP_WHITE_BALANCE_RED_V: "CAP_PROP_WHITE_BALANCE_RED_V",
    cv2.CAP_PROP_ZOOM: "CAP_PROP_ZOOM",
    cv2.CAP_PROP_FOCUS: "CAP_PROP_FOCUS",
    cv2.CAP_PROP_GUID: "CAP_PROP_GUID",
    cv2.CAP_PROP_ISO_SPEED: "CAP_PROP_ISO_SPEED",
    cv2.CAP_PROP_BACKLIGHT: "CAP_PROP_BACKLIGHT",
    cv2.CAP_PROP_PAN: "CAP_PROP_PAN",
    cv2.CAP_PROP_TILT: "CAP_PROP_TILT",
    cv2.CAP_PROP_ROLL: "CAP_PROP_ROLL",
    cv2.CAP_PROP_IRIS: "CAP_PROP_IRIS",
    cv2.CAP_PROP_SETTINGS: "CAP_PROP_SETTINGS",
    cv2.CAP_PROP_BUFFERSIZE: "CAP_PROP_BUFFERSIZE",
    cv2.CAP_PROP_AUTOFOCUS: "CAP_PROP_AUTOFOCUS",
    cv2.CAP_PROP_SAR_NUM: "CAP_PROP_SAR_NUM",
    cv2.CAP_PROP_SAR_DEN: "CAP_PROP_SAR_DEN",
    cv2.CAP_PROP_BACKEND: "CAP_PROP_BACKEND",
    cv2.CAP_PROP_CHANNEL: "CAP_PROP_CHANNEL",
    cv2.CAP_PROP_AUTO_WB: "CAP_PROP_AUTO_WB",
    cv2.CAP_PROP_WB_TEMPERATURE: "CAP_PROP_WB_TEMPERATURE",
    cv2.CAP_PROP_CODEC_PIXEL_FORMAT: "CAP_PROP_CODEC_PIXEL_FORMAT",
    cv2.CAP_PROP_BITRATE: "CAP_PROP_BITRATE",
    cv2.CAP_PROP_ORIENTATION_META: "CAP_PROP_ORIENTATION_META",
    cv2.CAP_PROP_ORIENTATION_AUTO: "CAP_PROP_ORIENTATION_AUTO",
    cv2.CAP_PROP_HW_ACCELERATION: "CAP_PROP_HW_ACCELERATION",
    cv2.CAP_PROP_HW_DEVICE: "CAP_PROP_HW_DEVICE",
    cv2.CAP_PROP_HW_ACCELERATION_USE_OPENCL: "CAP_PROP_HW_ACCELERATION_USE_OPENCL",
    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC: "CAP_PROP_OPEN_TIMEOUT_MSEC",
    cv2.CAP_PROP_READ_TIMEOUT_MSEC: "CAP_PROP_READ_TIMEOUT_MSEC",
    cv2.CAP_PROP_STREAM_OPEN_TIME_USEC: "CAP_PROP_STREAM_OPEN_TIME_USEC",
    cv2.CAP_PROP_VIDEO_TOTAL_CHANNELS: "CAP_PROP_VIDEO_TOTAL_CHANNELS",
    cv2.CAP_PROP_VIDEO_STREAM: "CAP_PROP_VIDEO_STREAM",
    cv2.CAP_PROP_AUDIO_STREAM: "CAP_PROP_AUDIO_STREAM",
    cv2.CAP_PROP_AUDIO_POS: "CAP_PROP_AUDIO_POS",
    cv2.CAP_PROP_AUDIO_SHIFT_NSEC: "CAP_PROP_AUDIO_SHIFT_NSEC",
    cv2.CAP_PROP_AUDIO_DATA_DEPTH: "CAP_PROP_AUDIO_DATA_DEPTH",
    cv2.CAP_PROP_AUDIO_SAMPLES_PER_SECOND: "CAP_PROP_AUDIO_SAMPLES_PER_SECOND",
    cv2.CAP_PROP_AUDIO_BASE_INDEX: "CAP_PROP_AUDIO_BASE_INDEX",
    cv2.CAP_PROP_AUDIO_TOTAL_CHANNELS: "CAP_PROP_AUDIO_TOTAL_CHANNELS",
    cv2.CAP_PROP_AUDIO_TOTAL_STREAMS: "CAP_PROP_AUDIO_TOTAL_STREAMS",
    cv2.CAP_PROP_AUDIO_SYNCHRONIZE: "CAP_PROP_AUDIO_SYNCHRONIZE",
    cv2.CAP_PROP_LRF_HAS_KEY_FRAME: "CAP_PROP_LRF_HAS_KEY_FRAME",
    cv2.CAP_PROP_CODEC_EXTRADATA_INDEX: "CAP_PROP_CODEC_EXTRADATA_INDEX",
    cv2.CAP_PROP_FRAME_TYPE: "CAP_PROP_FRAME_TYPE",
    cv2.CAP_PROP_N_THREADS: "CAP_PROP_N_THREADS",
    cv2.CAP_PROP_PTS: "CAP_PROP_PTS",
    cv2.CAP_PROP_DTS_DELAY: "CAP_PROP_DTS_DELAY",
}

VIDEO_CAPTURE_PROPERTIES_INVERSE: dict[str, int] = {
    VIDEO_CAPTURE_PROPERTIES[cap_prop]: cap_prop
    for cap_prop in VIDEO_CAPTURE_PROPERTIES
}


def cam_print_props(cam: cv2.VideoCapture):
    for prop, name in VIDEO_CAPTURE_PROPERTIES.items():
        value = cam.get(prop)
        if value > 0:
            print(f"{name}: {value}")


def cam_get_effective_frame_size(cam_id: Union[int, str], props: dict):
    cam = cv2.VideoCapture(cam_id)
    h_w_fps = cam_set_props(cam, props)
    cam.release()
    return h_w_fps


def cam_set_props(cam: cv2.VideoCapture, props: dict):
    if props is None:
        props = {}

    if "CAP_PROP_FRAME_SIZE" in props:
        h, w = props["CAP_PROP_FRAME_SIZE"]
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    elif "CAP_PROP_FRAME_WIDTH" in props and "CAP_PROP_FRAME_HEIGHT" in props:
        h = props["CAP_PROP_FRAME_HEIGHT"]
        w = props["CAP_PROP_FRAME_WIDTH"]
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, h)
    if "CAP_PROP_FPS" in props:
        cam.set(cv2.CAP_PROP_FPS, props["CAP_PROP_FPS"])

    for cap_prop in props:
        if cap_prop in ["CAP_PROP_FRAME_SIZE", "CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT"]:
            continue
        if cap_prop not in VIDEO_CAPTURE_PROPERTIES_INVERSE:
            continue
        cap_id = VIDEO_CAPTURE_PROPERTIES_INVERSE[cap_prop]
        cap_value = props[cap_prop]
        cam.set(cap_id, cap_value)
    # end

    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    return h, w, fps


def cam_get_frame_size(cam: cv2.VideoCapture) -> tuple[int, int]:
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    return h, w


def cam_get_frame_info(cam: cv2.VideoCapture) -> tuple[float, float]:
    frames = cam.get(VIDEO_CAPTURE_PROPERTIES_INVERSE["CAP_PROP_POS_FRAMES"])
    msec = cam.get(VIDEO_CAPTURE_PROPERTIES_INVERSE["CAP_PROP_POS_MSEC"])
    return int(frames), int(msec + 0.5)


def frame_resize(frame: np.ndarray, props: dict) -> np.ndarray:
    if props is None:
        return frame
    h, w = 0, 0
    if "CAP_PROP_FRAME_SIZE" in props:
        h, w = props["CAP_PROP_FRAME_SIZE"]
    elif "CAP_PROP_FRAME_WIDTH" in props and "CAP_PROP_FRAME_HEIGHT" in props:
        w = props["CAP_PROP_FRAME_WIDTH"]
        h = props["CAP_PROP_FRAME_HEIGHT"]

    fh, fw, fc = frame.shape
    if w != 0 and h != 0 and (fw > w or fh > h):
        frame = cv2.resize(frame, (w, h))
    return frame


# ---------------------------------------------------------------------------
# VideoCaptureForeground
# ---------------------------------------------------------------------------

class VideoCaptureForeground:
    def __init__(self, cam_id, props):
        cam = cv2.VideoCapture(cam_id)
        cam_set_props(cam, props)
        self.cam = cam
        self.props = props
        self.frame_id = 0

    def release(self):
        self.cam.release()
    # end

    def read(self):
        ret = False
        frame = None

        while not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            ret, frame = self.cam.read()

        frame_dt = time.time()
        frame_id = self.frame_id
        frame = frame_resize(frame, self.props)

        self.frame_id += 1

        return frame, frame_dt, frame_id
# end


# ---------------------------------------------------------------------------
# VideoCaptureThread
# ---------------------------------------------------------------------------

def video_capture_thread(
    cam_id,
    status: list[int],
    frame_slot: list[np.ndarray],
    frame_dt: list[float],
    frame_id_slot: list[int],
    props
):
    print("video_capture_thread started")
    # status
    #   0 -> continue
    #   1 -> take a frame
    #   2 -> terminate
    cam = cv2.VideoCapture(cam_id)
    cam_set_props(cam, props)

    start_time = time.time()
    check_time = start_time
    frame_id = 0

    # read, grab, retrieve
    while status[0] != 2:
        ret, frame = cam.read()
        if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue

        if status[0] == 1:
            # print("... ... grab frame")
            frame_slot[0] = frame
            frame_dt[0] = time.time()
            frame_id_slot[0] = frame_id
            status[0] = 0
        # end
        frame_id += 1

        now = time.time()
        if (now - check_time) > 3:
            check_time = now
            delta = now - start_time
            print(f"...  {frame_id:5}: {frame_id / delta:.4} fps")
    # end

    cam.release()
    print("video_capture_thread terminated:", status[0])


class VideoCaptureThread:

    def __init__(self, cam_id, props=None):
        self.props = props
        self.status = [0]
        self.frame: list[np.ndarray] = [None]
        self.frame_dt: list[float] = [0.0]
        self.frame_id: list[int] = [0]

        self.thread_video_capture = Thread(
            target=video_capture_thread,
            kwargs={
                "cam_id": cam_id,
                "status": self.status,
                "frame": self.frame,
                "frame_dt": self.frame_dt,
                "frame_id": self.frame_id,
                "props": props
            })
        self.thread_video_capture.start()

        pass

    def release(self):
        self.status[0] = 2
        self.thread_video_capture.join()

    def read(self):
        self.status[0] = 1
        while self.status[0] != 0:
            time.sleep(1)

        frame = self.frame[0]
        frame_dt = self.frame_dt[0]
        frame_id = self.frame_id[0]

        frame = frame_resize(frame, self.props)

        return frame, frame_dt, frame_id
# end


# ---------------------------------------------------------------------------
# VideoCaptureProcess
# ---------------------------------------------------------------------------

def video_capture_process(
        cam_id,
        status: Value,
        queue: Queue,
        props: dict
):
    print("video_capture_process started")
    # status
    #   0 -> continue
    #   1 -> take a frame
    #   2 -> terminate
    cam = cv2.VideoCapture(cam_id)
    cam_set_props(cam, props)
    # cam_print_props(cam)

    start_time = time.time()
    check_time = start_time
    frame_id = 0

    # read, grab, retrieve
    while status.value != 2:
        ret, frame = cam.read()
        if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue

        if status.value == 1:
            # print("... ... grab frame")
            # queue.put((fid, frame), False)
            frame_dt = time.time()
            queue.put((frame, frame_dt, frame_id))
            status.value = 0

        frame_id += 1

        # now = time.time()
        # if (now - check_time) > 3:
        #     check_time = now
        #     delta = now - start_time
        #     fid, fdt = cam_get_frame_info(cam)
        #     print(f"...  {frame_id:5}: {frame_id / delta:.4} fps")
        #     print(f"...  {fid:5}: {fdt:6} fps")
    # end
    cam.release()
    cv2.destroyAllWindows()
    print("video_capture_process terminated:", status.value)


class VideoCaptureProcess:
    def __init__(self, cam_id, props=None):

        h, w, fps = cam_get_effective_frame_size(cam_id, props)

        self.cam_id = cam_id
        self.props = props
        self.fps = fps

        self.status = Value('b', 0)
        self.queue = SimpleQueue()

        self.process_video_capture = Process(
            target=video_capture_process,
            args=(cam_id, self.status, self.queue, props)
        )
        self.process_video_capture.start()

        while not self.process_video_capture.is_alive():
            time.sleep(0.1)
        pass

    def release(self):
        self.status.value = 2
        self.process_video_capture.join()

    # end

    def read(self):
        self.status.value = 1

        frame, frame_dt, frame_id = self.queue.get()

        frame = frame_resize(frame, self.props)

        return frame, frame_dt, frame_id
    # end
# end

