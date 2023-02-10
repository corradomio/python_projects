import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoTokenizer, AutoModel
from random import shuffle
from scipy.spatial.distance import pdist, squareform
from pprint import pprint


def shuffle_list(l):
    shuffle(l)
    return l


def to_numpy(tensor):
    return tensor[0].detach().mean(1)[0].numpy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

nl_tokens = tokenizer.tokenize("return maximum value")
code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens+[tokenizer.eos_token]
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)


tokens = tokenizer.tokenize("def shuffle_list(l): shuffle(l); return l")
tokens_two = tokenizer.convert_tokens_to_ids(tokens)

context_embeddings = model(torch.tensor(tokens_ids)[None, :])
# print(context_embeddings)

embs = np.array([
    to_numpy(model(torch.tensor(tokens_ids)[None, :])),
    # to_numpy(model(torch.tensor(shuffle_list(tokens_ids))[None, :])),
    to_numpy(model(torch.tensor(shuffle_list(tokens_ids))[None, :])),
    to_numpy(model(torch.tensor(shuffle_list(tokens_ids))[None, :])),

    to_numpy(model(torch.tensor(tokens_two)[None, :]))
])

dist = squareform(pdist(embs, metric='cosine'))
pprint(dist)



