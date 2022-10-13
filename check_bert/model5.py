from pprint import pprint
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/bert-large-nli-cls-token')

# Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence']

# Sentences are encoded by calling model.encode()
embedding1 = model.encode(sentence)
pprint(embedding1)

# Sentences we want to encode. Example:
sentence = ['framework','generates','embeddings','for','each','input','sentence','This']
embedding2 = model.encode(sentence)
print("--")
pprint(embedding2)
print("--")
pprint(embedding2-embedding1)
