from datasets import list_datasets
all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currently available on the Hub")
print(f"The first 10 are: {all_datasets[:10]}")

from datasets import load_dataset
emotions = load_dataset("emotion")
print(emotions)
