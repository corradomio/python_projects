from dalle2 import Dalle2

dalle = Dalle2("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")   # your bearer key

file_paths = dalle.generate_and_download("portal to another dimension, digital art")
print(file_paths)
