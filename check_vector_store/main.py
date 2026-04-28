from pathlib import Path
from vector_store import VectorStore
from PIL import Image
import numpy as np

subdirs = ["face", "face_recognition", "random_crop"]

def main():

    vs = VectorStore()

    tmp_result = Path(r"D:\Projects.ebtic\project.diwang\lab_monitoring\tmp_result")
    for d in tmp_result.iterdir():
        for s in subdirs:
            dir = d / s
            if not dir.exists(): continue
            for img_path in dir.glob("*.jpg"):
                pic = Image.open(img_path)
                pix = np.array(pic)
                vs.put(str(img_path), pix)




if __name__ == "__main__":
    main()