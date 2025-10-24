from pprint import pprint
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("open-thoughts/OpenThoughts-114k", "default")
# ds = load_dataset("bespokelabs/Bespoke-Stratos-17k")

ds = load_dataset("isaiahbjork/chain-of-thought")

print(ds["train"])

for e in ds["train"]:
    pprint(e)
    break


from datasets import load_dataset


