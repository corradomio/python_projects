import jsonx
from datasets import list_datasets

'huggingface_hub.hf_api.DatasetInfo'

all_datasets = list_datasets(with_details=True)
ten_datasets = all_datasets[:10]

# with open("datasets.json", mode="w") as fp:
#     json.dump(all_datasets, fp, indent=4)

jsonx.write(all_datasets, "datasets.json")


# ten_datasets = all_datasets[:10]
#
# print(len(all_datasets))
# for ds in all_datasets:
#     print(ds)


