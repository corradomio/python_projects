from pprint import pprint
from transformers import BloomConfig, BloomModel

# Initializing a Bloom configuration
configuration = BloomConfig()

# Initializing a model (with random weights) from the configuration
model = BloomModel(configuration)

# Accessing the model configuration
configuration = model.config

pprint(configuration)


from transformers import AutoTokenizer, BloomModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomModel.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Ciao come stai", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pprint(last_hidden_states)
