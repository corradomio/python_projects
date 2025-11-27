# https://github.com/Ikomia-dev/notebooks/blob/main/examples/HOWTO_use_DeepSORT_with_Ikomia_API.ipynb
#
# https://www.labellerr.com/blog/deepsort-real-time-object-tracking-guide/
# https://github.com/nwojke/deep_sort?ref=labellerr.com
#
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
import numpy as np
import random
import clip


def deepsort(path, output='output.mp4', target_classes=None):
    # Initialize YOLOv10 model
    model = YOLO('yolov10n.pt')  # Choose your model

    # Initialize video capture
    cap = cv2.VideoCapture(path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output directory if not exists
    os.makedirs("output_videos", exist_ok=True)
    output_path = f"output_videos/{output}"

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_age=20,
        n_init=2,
        embedder='clip_ViT-B/16',
        half=True,
        embedder_gpu=True
    )

    # Create color palette for IDs
    color_palette = {}

    # Set default target classes (person, car, truck) if none provided
    if target_classes is None:
        target_classes = [0, 2, 7]  # COCO class IDs: 0=person, 2=car, 7=truck

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv10 detection
            results = model(frame, verbose=False)[0]

            # Convert detections to DeepSort format
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Filter by target classes
                if cls_id in target_classes:
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Draw tracking results
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Generate unique color for each ID
                if track_id not in color_palette:
                    # Generate random but distinct color
                    color_palette[track_id] = (
                        random.randint(50, 200),
                        random.randint(50, 200),
                        random.randint(50, 200)
                    )
                color = color_palette[track_id]

                # Draw thicker bounding box (4px instead of 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                # Create white background for ID text
                text = f"ID:{track_id}"
                text_scale = 1.5  # Increased from 0.7 (3x larger)
                text_thickness = 4
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                            text_scale, text_thickness)[0]

                # Position background above bounding box
                bg_x1 = x1
                bg_y1 = max(0, y1 - text_size[1] - 10)  # Ensure within frame
                bg_x2 = x1 + text_size[0] + 5
                bg_y2 = y1 - 10

                # Draw background if it's within frame boundaries
                if bg_y1 >= 0 and bg_y2 < frame_height and bg_x2 < frame_width:
                    cv2.rectangle(frame,
                                  (bg_x1, bg_y1),
                                  (bg_x2, bg_y2),
                                  (255, 255, 255), -1)  # White background

                    # Display ID with same color as bounding box
                    cv2.putText(frame, text, (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, text_scale, color,
                                text_thickness)

            # Write frame to video file
            out.write(frame)

            # Print progress
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release resources
        cap.release()
        out.release()
        print(f"Video saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")
