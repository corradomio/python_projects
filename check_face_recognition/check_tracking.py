from ultralytics import YOLO

print("yolo26n.pt")
# Load an official or custom model
model = YOLO("yolo26n.pt")  # Load an official Detect model
# model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo26n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom-trained model

print("track")
# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack

# results = model.track(
#     r"E:\Movies\FILM - Polar - 2019.iTALiAN.WEBRiP.XviD-PRiME.avi",
#     show=True,
#     tracker="bytetrack.yaml",
#     verbose=False
# )  # Tracking with default tracker

results = model.track(
    # r"E:\Movies\FILM - Polar - 2019.iTALiAN.WEBRiP.XviD-PRiME.avi",
    # r"E:\Dropbox\Movies\Scifi Movies\Tron.Ares.2025.bdrip.1080p.x264.ita.eac3.ac3.eng.ac3.subs.fd.mkv",
    0,
    show=True,
    tracker="botsort.yaml",
    verbose=False
)