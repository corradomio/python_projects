#
# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
#


import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
from pprint import pprint

sentences = ["I ate dinner.",
             "We had a three-course meal.",
             "Brad came to dinner with us.",
             "He loves fish tacos.",
             "In the end, we all felt like we ate too much.",
             "We all agreed; it was a magnificent evening."]

# Tokenization of each document
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))
print(tokenized_sent)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# import
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
pprint(tagged_data)

## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)

'''
vector_size = Dimensionality of the feature vectors.
window = The maximum distance between the current and predicted word within a sentence.
min_count = Ignores all words with total frequency lower than this.
alpha = The initial learning rate.
'''

## Print model vocabulary
pprint(model.wv.vocab)

test_doc = word_tokenize("I had pizza and pasta".lower())
test_doc_vector = model.infer_vector(test_doc)
model.docvecs.most_similar(positive=[test_doc_vector])

'''
positive = List of sentences that contribute positively.
'''

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model.encode(sentences)

# print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
# print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])

query = "I had pizza and pasta"
query_vec = model.encode([query])[0]

for sent in sentences:
    sim = cosine(query_vec, model.encode([sent])[0])
    print("Sentence = ", sent, "; similarity = ", sim)


from models import InferSent
import torch

V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'GloVe/glove.840B.300d.txt'
model.set_w2v_path(W2V_PATH)

model.build_vocab(sentences, tokenize=True)

query = "I had pizza and pasta"
query_vec = model.encode(query)[0]
pprint(query_vec)

similarity = []
for sent in sentences:
    sim = cosine(query_vec, model.encode([sent])[0])
    print("Sentence = ", sent, "; similarity = ", sim)
