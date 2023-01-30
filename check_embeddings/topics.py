import numpy as np

class GensimTopics:

    def __init__(self, dictionary, topics):
        self.dictionary = dictionary
        # d.id2token
        # d.token2id
        self.topics = topics
        n_topics = len(topics)
        n_words = len(dictionary.id2token)
        w_weights = np.zeros((n_topics, n_words))
        t_weights = np.zeros(n_topics)
        # word/token weights
        self.w_weights = w_weights
        # topic weights
        self.t_weights = t_weights

        for i in range(n_topics):
            topic = topics[i]
            t_weights[i] = topic[1]
            for w, weight in topic[0]:
                j = dictionary.id2token[w]
                w_weights[i, j] = weight
            # end
        # end
    # end

    @property
    def token_weights(self):
        return self.w_weights

