import torch
import torchreid
import matplotlib.pyplot as plt
from path import Path

from torchreidx.utils import FeatureExtractor
from torchreid.metrics.distance import compute_distance_matrix


def main():
    names = list(torchreid.models.__model_factory.keys())

    for name in names:
        print("--", name, "--")
        try:
            extractor = FeatureExtractor(
                model_name=name,
                # model_path='a/b/c/model.pth.tar',
                device='cuda'
            )

            root_dir = r"D:\Projects.ebtic\project.diwang\lab_monitoring\tmp_result\0_0_DONE\random_crop"
            other_dir = r"D:\Projects.ebtic\project.diwang\lab_monitoring\tmp_result\0_1_DONE\random_crop"
            # image_list = root_dir.files("*.jpg")

            embeddings: torch.Tensor = extractor(root_dir)
            embother: torch.Tensor = extractor(other_dir)
            print(embeddings.shape)


            edist: torch.Tensor = compute_distance_matrix(embeddings, embother, metric="euclidean")
            plt.title(f"{name}/euclidean")
            plt.imshow(edist.detach().cpu().numpy())
            plt.savefig(f"images/{name}-euclidean.jpg", dpi=300)

            cdist: torch.Tensor = compute_distance_matrix(embeddings, embother, metric="cosine")
            plt.title(f"{name}/cosine")
            plt.imshow(cdist.detach().cpu().numpy())
            plt.savefig(f"images/{name}-cosine.jpg", dpi=300)

            # for i in range(n):
            #     for j in range(i+1,n):
            #         edist = compute_distance_matrix(embeddings, embeddings, metric="euclidean")
            #         cdist = compute_distance_matrix(embeddings, embeddings, metric="cosine")
            #         pass
        except Exception as e:
            print("... ERROR:", e)
    # end
    pass


if __name__ == "__main__":
    main()
