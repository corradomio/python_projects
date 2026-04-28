from multiprocessing import freeze_support
import time

import cv2
import opencvx as cvx

def timestamp():
    return time.time()

#
#         "#camera_ip": [
#             "rtsp://admin:password1234@10.248.37.111/Streaming/Channels/101?tcp",
#             "rtsp://admin:password1234@10.248.37.108/Streaming/Channels/101?tcp",
#             "rtsp://admin:password1234@10.248.37.12/Streaming/Channels/101?tcp",
#             "rtsp://admin:password1234@10.248.37.100/Streaming/Channels/101?tcp"
#         ]
#

def main():
    # Open the default camera
    print("start camera in background")
    # cam = cvx.VideoCaptureProcess3(
    #     0,
    #     # "rtsp://admin:password1234@10.248.37.111/Streaming/Channels/101?tcp",
    #     {
    #         "CAP_PROP_FRAME_SIZE": (576, 1024),
    #         "CAP_PROP_FPS": 25
    #     })
    # cam = cvx.VideoCaptureForeground(
    #     0,
    #     # "rtsp://admin:password1234@10.248.37.111/Streaming/Channels/101?tcp",
    #     {
    #         "CAP_PROP_FRAME_SIZE": (576, 1024),
    #         "CAP_PROP_FPS": 25
    #     })
    # cam = cvx.VideoCaptureThread(
    #     0,
    #     # "rtsp://admin:password1234@10.248.37.111/Streaming/Channels/101?tcp",
    #     {
    #         "CAP_PROP_FRAME_SIZE": (576, 1024),
    #         "CAP_PROP_FPS": 25
    #     })
    cam = cvx.VideoCaptureProcess(
        0,
        # "rtsp://admin:password1234@10.248.37.111/Streaming/Channels/101?tcp",
        {
            "CAP_PROP_FRAME_SIZE": (576, 1024),
            "CAP_PROP_FPS": 25
        })

    # Get the default frame width and height

    # print("wait")
    # time.sleep(10000)

    start_time = time.time()
    log_time = start_time
    main_id = 0
    while True:
        # time.sleep(1)

        frame, frame_dt, frame_id = cam.read()

        now = time.time()
        if now - log_time > 3:
            log_time = now
            delta = (timestamp() - start_time)
            print(f"main: {main_id :4}: {main_id/delta :.4} fps")
            print(f" cam: {frame_id:4}: {frame_id/delta:.4} fps")

        main_id += 1

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    # end

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    freeze_support()
    main()