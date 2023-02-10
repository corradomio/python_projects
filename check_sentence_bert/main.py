import numpy as np
from random import shuffle
from pprint import pprint
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform


model = SentenceTransformer('all-MiniLM-L6-v2')


def shuffle_sentence(s):
    words = s.split(' ')
    shuffle(words)
    t = " ".join(words)
    return t


sentence = 'Space: final frontier. These are the voyages of the starship Enterprise. ' \
           'His mission is to explore strange new worlds in search of new life forms ' \
           'and new civilizations to boldly go where no one has gone before'

# sentence = 'the cat eats the fish'

# Our sentences we like to encode
sentences = [
    sentence,
    shuffle_sentence(sentence),
    shuffle_sentence(sentence),
    # shuffle_sentence(sentence),
    shuffle_sentence(sentence),
    'Sorting algorithm. The default is ‘quicksort’. Note that both ‘stable’ and ‘mergesort’ use timsort or radix sort under the covers and, in general, the actual implementation will vary with data type. The ‘mergesort’ option is retained for backwards compatibility.'

    # 'the cat eats the fish',
    # 'the fish eats the cat',
    # 'the the cat cat eats eats the the fish fish',
    # 'the can eats the fish',
]


# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

embeddings = np.sort(embeddings, axis=1)

dist = squareform(pdist(embeddings))
pprint(dist)


# Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")
