import cv2
from ultralytics import YOLO


model = YOLO("yolo-Weights/yolov8n.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


def print_results(results):
    for r in results:
        # Access bounding box coordinates (e.g., xyxy format: [x_min, y_min, x_max, y_max])
        boxes = r.boxes.xyxy.cpu().numpy()  # Convert to NumPy array for easier handling

        # Access class IDs
        class_ids = r.boxes.cls.cpu().numpy()

        # Access class names
        class_names = r.names

        # Loop through each detection
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            class_id = int(class_ids[i])
            class_name = class_names[class_id]

            print(f"Detected: {class_name}")
            print(f"Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        # end
    # end
# end


def main():

    # while True:
    #     ret, img= cap.read()
    #     cv2.imshow('Webcam', img)
    #
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    c = 0
    while True:
        ret, frame = cap.read()

        if c < 200:
            c += 1
            print(".", end="")
        else:
            c = 0
            print(".")
            results = model(frame, stream=True)
            print_results(results)

        # results = model(frame, stream=True)
        # print(results)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# end

if __name__ == "__main__":
    main()
