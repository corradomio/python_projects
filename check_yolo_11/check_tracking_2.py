from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolo11n.pt")
results = model.track(source=
                      # "https://youtu.be/LNwODJXcvt4"
                      # "https://youtu.be/bON9BZ2t2SM"
                      "https://youtu.be/_XedA4_8k5s"
                      , conf=0.3, iou=0.5, show=True)

