facenet-pytorch
Pretrained Pytorch face detection (MTCNN)
facial recognition (InceptionResnet)
RetinaFace


face_detection
    https://github.com/hukkelas/DSFD-Pytorch-Inference


Face direction:
    - mapping these 2D points to a 3D model to calculate yaw, pitch, and roll angles
      Perspective-n-Point (PnP) solver (OpenCV: solvePnP)
      https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
    - 6DRepNet
      https://github.com/thohemp/6DRepNet
      weights
      https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth

    roll, pitch, yaw / euler angles
        https://simple.wikipedia.org/wiki/Pitch,_yaw,_and_roll
        https://en.wikipedia.org/wiki/Euler_angles

        X, Roll: Controls banking, managed by ailerons.
        Y, Pitch: Controls nose-up or nose-down attitude, managed by elevators.
        Z, Yaw: Controls heading, often managed by a rudder in aircraft.

        euler angles to rotation matrix
        available in

    - FSA-Net (Fine-Grained Structure Aggregation Network)
    - HopeNet
    - DirectMHP (Direct Multi-Hypothesis Head Pose)
    - TriNet / FDN  ???
    - 6DRepNet360
    -

MTCNN:
    boxes/landmarks=False -> [[tx, ty, bx, by], ...]  top-left
    boxes/landmarks=True  ->
    (
        [[tx, ty, bx, by], ...],
        [accuracy1, ...],
        [   [ [x1, y1], [x2,y2] [x3, y3], x4, y4], [x5, y5] ]
            , ...
        ]
    )

    0) eye right,
    1) eye left,
    2) nose,
    3) mouth right,
    4) mouth left


Medimum: facenet-pytorch | Pretrained Pytorch face detection (MTCNN) and facial recognition (InceptionResnet) models | Computer Vision library
    https://medium.com/@danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144


osnet_x1_0