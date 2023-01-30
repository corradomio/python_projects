import numpy as np
import lda

# corpus = [[(i*10+j, 1) for j in range(10)] for i in range(10)]
# id2word = {i: str(i) for i in range(100)}

def corpus_matrix():
    X = np.zeros((10, 100), dtype=int)
    for i in range(10):
        for j in range(10):
            X[i, i*10+j] = 1
    return np.array(X), tuple(str(i) for i in range(100))

X, vocab = corpus_matrix()



import lda.datasets


# X = lda.datasets.load_reuters()
# vocab = lda.datasets.load_reuters_vocab()
print(X.shape)
print(X.sum())

model = lda.LDA(n_topics=10, n_iter=1000, random_state=1)
model.fit(X)
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))