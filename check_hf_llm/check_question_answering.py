from datasets import get_dataset_config_names, load_dataset

domains = get_dataset_config_names("subjqa")
print(domains)


subjqa = load_dataset("subjqa", name="electronics")
print(subjqa["train"]["answers"][1])
