from dalle2 import Dalle2

dalle = Dalle2("sess-1Otsq3r3zPoGpB2CGbb1XPeRc6EFpUjmBqhf47z0")   # your bearer key

file_paths = dalle.generate_and_download("portal to another dimension, digital art")
print(file_paths)
