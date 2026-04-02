https://viso.ai/computer-vision/deepface/

https://github.com/serengil/deepface

build_model
    models:
        VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace GhostFaceNet and Buffalo_L
            for face recognition
        Age, Gender, Emotion, Race
            for facial attributes
        opencv, mtcnn, ssd, dlib, retinaface, mediapipe,
                    yolov8n, yolov8m, yolov8l, yolov11n, yolov11s, yolov11m,
                    yolov11l, yolov12n, yolov12s, yolov12m, yolov12l,
                    yunet, fastmtcnn or centerface
            for face detectors
    tasks
        facial_recognition, facial_attribute, face_detector, spoofing


build_index()


verify()
    Face Verification

    This function determines whether two facial images belong to the same person or to different individuals.
    The function returns a dictionary, where the key of interest is verified: True indicates the images are of the same
    person, while False means they are of different people.

find()
    Face recognition
    Face recognition requires applying face verification many times. DeepFace provides an out-of-the-box find function
    that searches for the identity of an input image within a specified database path.

analyze()
    Facial Attribute Analysis
    DeepFace also comes with a strong facial attribute analysis module including age, gender, facial expression
    (including angry, fear, neutral, sad, disgust, happy and surprise) and race (including asian, white, middle
    eastern, indian, latino and black) predictions.

stream()
    Real Time Analysis
    You can run deepface for real time videos as well. Stream function will access your webcam and apply both face
    recognition and facial attribute analysis. The function starts to analyze a frame if it can focus a face
    sequentially 5 frames. Then, it shows results 5 seconds.

represent()
    Embeddings - Tutorial, Demo
    Face recognition models basically represent facial images as multi-dimensional vectors. Sometimes, you need those
    embedding vectors directly. DeepFace comes with a dedicated representation function.

extract_faces()

detectFace()

register()

search()



FaceNet128
    InceptionResNetV1(128)