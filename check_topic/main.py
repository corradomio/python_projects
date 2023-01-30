from pprint import pprint
# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 1000
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
# temp = dictionary[0]  # This is only to "load" the dictionary.
# id2word = dictionary.id2token

corpus = [[(i*10+j, 1) for j in range(10)] for i in range(10)]
id2word = {i: str(i) for i in range(100)}

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus)

for i in range(10):
    print(model.get_topic_terms(i, 100))
