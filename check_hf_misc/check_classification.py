from huggingface_hub import list_datasets
from datasets import load_dataset, DatasetInfo

all_datasets = list_datasets()
# from datasets import list_datasets
# all_datasets = list_datasets()

for ds in all_datasets:
    print(ds.id)
# print(f"There are {len(all_datasets)} datasets currently available on the Hub")
# print(f"The first 10 are: {all_datasets[:10]}")

emotions = load_dataset("emotion")
print(emotions)
