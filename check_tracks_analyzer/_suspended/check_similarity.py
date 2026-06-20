from scipy.spatial.distance import cdist

from human.clipreid import ClipReID


def similarity(image1, image2) -> float:
    emb1 = ClipReID.represent(image1, "DukeMTMC__cnn_base").reshape((1,-1))
    emb2 = ClipReID.represent(image2, "DukeMTMC__cnn_base").reshape((1,-1))
    return 1 - cdist(emb1,emb2, metric="cosine").max()


def main():
    print(similarity(
        r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-08\20260513_155653\2_9457_DONE\random_crop\20260608_142304_crop_no_margin.jpg",
        r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-08\20260513_155653\1_7326_DONE\random_crop\20260608_142241_crop_no_margin.jpg"
    ))

    pass


if __name__ == "__main__":
    main()

