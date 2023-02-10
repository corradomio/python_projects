from typing import Optional

import numpy as np
import tomotopy as tp

from .loggingx import Logger
from .utils_numpy import is_matrix
from .utils_stdlib import is_string_list, flatten


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def topic_names(words: list[str], topic_distrib: np.ndarray) -> list[str]:
    n, m = topic_distrib.shape
    tnames: list[str] = []
    twlist = []
    #
    # retrieve the list of words fore each topic ordered by probability (desc)
    #
    for i in range(n):
        # [(w, prob), ...]
        twords = map(lambda j: (words[j], topic_distrib[i, j]), range(m))
        # order by inverse prob
        twords = sorted(twords, key=lambda wd: wd[1], reverse=True)
        # [w, ...]
        twords = list(map(lambda wd: wd[0], twords))
        # [[w11, ...], ...]
        twlist.append(twords)
    # end
    #
    # for each topic select a number of words such that make the name unique
    # for some topics it is enough a single word, for other 2 or more
    #
    for i in range(n):
        retry = True
        p = 0
        while retry:
            retry = False
            p += 1
            iprefix = twlist[i][0:p]
            for j in range(n):
                if i == j: continue
                jprefix = twlist[j][0:p]
                if iprefix == jprefix:
                    retry = True
                    break
            # end
            if retry:
                continue
            else:
                break

        # create the topic name
        tname = "t_" + "_".join(twlist[i][0:p])
        tnames.append(tname)
    # end
    return tnames
# end


def topic_tokens_order(topic_distrib: np.ndarray) -> np.ndarray:
    n, m = topic_distrib.shape
    twlist = []
    for i in range(n):
        # [(w, prob), ...]
        twords = map(lambda j: (j, topic_distrib[i, j]), range(m))
        # order by inverse prob
        twords = sorted(twords, key=lambda wd: wd[1], reverse=True)
        # [w, ...]
        twords = list(map(lambda wd: wd[0], twords))
        # [[w11, ...], ...]
        twlist.append(twords)
    # end
    tw = np.array(twlist, dtype=int)
    return tw
# end


def document_topics_order(topic_distrib: np.ndarray) -> np.ndarray:
    n, m = topic_distrib.shape
    dtlist = []
    for i in range(n):
        # [(t, prob), ...]
        dtopics = map(lambda j: (j, topic_distrib[i, j]), range(m))
        # order by inverse prob
        dtopics = sorted(dtopics, key=lambda dt: dt[1], reverse=True)
        # [t, ...]
        dtopics = list(map(lambda dt: dt[0], dtopics))
        # [[t11, ...], ...]
        dtlist.append(dtopics)
    # end
    dt = np.array(dtlist, dtype=int)
    return dt
# end


# ---------------------------------------------------------------------------
# LabelsDistributions
# ---------------------------------------------------------------------------

class LabelsDistribution:
    def __init__(self, labels: list[str], distributions: np.ndarray):
        assert is_string_list(labels)
        assert is_matrix(distributions)
        self._labels = labels
        self._distributions = distributions

    @property
    def labels(self):
        return self._labels

    @property
    def distributions(self):
        return self._distributions

    def sorted_distrib(self, i, top=0, label_only=False):
        # retrieve the labels distribution for the entity 'i'
        n, m = self.distributions.shape
        distib = self.distributions[i]
        # compose the list [(i, prob(i)), ...]
        pairs = [(i, distib[i]) for i in range(m)]
        # sort the list in desc order on prob
        pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
        # retrieve the list of labels based on the order
        if label_only:
            labels = [self.labels[pairs[i][0]] for i in range(m)]
        else:
            labels = [(self.labels[pairs[i][0]], pairs[i][1]) for i in range(m)]

        # select top labels if top is >= 1
        if top >= 1:
            labels = labels[0:top]
        # select enough labels to have sum(prob) >= top
        elif 0 < top < 1:
            p = 0
            for i in range(m):
                p += pairs[i][1]
                if p >= top:
                    break
            labels = labels[0:i + 1]
        # end

        # collect the list of labels
        return labels

    def sorted_distributions(self, top=0, label_only=False):
        n, m = self.distributions.shape
        slabels = []
        for i in range(n):
            labels = self.sorted_distrib(i, top=top, label_only=label_only)
            # collect the list of labels
            slabels.append(labels)
        # end
        return slabels
    # end
# end


# ---------------------------------------------------------------------------
# LDATopicModel
# ---------------------------------------------------------------------------

class LDATopicModel:
    def __init__(self, k: int = 0,
                 min_topics: int = 10,
                 patience: int = 10,
                 steps: int = 10):
        # k = 0 -> find the best k based on the 'perplexity
        self.k = k
        self.min_topics = min_topics
        self.patience = patience
        self.steps = steps

        self._mdl = None
        self._docids = []
        self._perplexity = []

        self._tokens = None
        self._topic_names = None
        self._topics = None
        self._documents = None
        self._file_names = None

        self._log = Logger.getLogger("LDATopicModel")
    # end

    def _clear(self):
        self._mdl = None
        self._docids = []
        self._perplexity = []

        self._tokens = None
        self._topics = None
        self._topic_names = None
        self._documents = None
        self._file_names = None
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def perplexity_history(self) -> np.ndarray:
        """
        List of perplexity computed in each step to find the best number of topics
        It is the
        """
        n = len(self._perplexity)
        pairs = []
        for i in range(n):
            if self._perplexity[i] != 0:
                pairs.append([i+0., self._perplexity[i]])
        return np.array(pairs)
    # end

    @property
    def tokens(self) -> list[str]:
        """List of tokens used in topics"""
        return self._tokens
    # end

    @property
    def topics(self) -> np.ndarray:
        """
        Matrix (n_topics, n_tokens)
        Each row is a topic and each column is the 'probability' of the token in the topic.
        The rows are normalized (sum = 1)
        :return: matrix (n_topics, n_tokens)
        """
        return self._topics
    # end

    @property
    def documents(self) -> np.ndarray:
        """
        Matrix (n_docs, n_topics)
        Each row is a document and each column is the probability of the topic in the document
        The rows are normalized (sum = 1)
        :return: matrix (n_docs, n_topics)
        """
        return self._documents
    # end

    @property
    def topic_names(self) -> list[str]:
        """
        Assign to each topic a name based on top tokens (tokens with highest 'probability')
        If two topics have the same 'k' tokens (in the same order), it is used the 'k+1' token,
        until the names became different
        :return: list of n_topics strings
        """
        return self._topic_names
    # end

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def fit(self, corpus: list[list[str]], names: Optional[list[str]] = None):
        self._clear()
        n = len(corpus)

        self._file_names = names if names is not None else list(map(str, range(1, n + 1)))

        self._log.info(f"fit with {n} documents")
        if self.k == 0:
            self.k = self._find_best_k(corpus)

        self._train_model(corpus, self.k)
        self._compute_distributions()
        self._log.info(f"done [{self.k}/{self._mdl.perplexity}]")
    # end

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    def _find_best_k(self, corpus):
        k = self.min_topics
        n = len(set(flatten(corpus)))
        self._set_perplexity(n, 0)

        mdl = self._train_model(corpus, k)
        ref_perplexity = self._set_perplexity(k, mdl.perplexity)
        steps = self.steps
        while steps > 2:
            k += steps
            mdl = self._train_model(corpus, k)
            perplexity = self._set_perplexity(k, mdl.perplexity)
            if perplexity < ref_perplexity:
                ref_perplexity = perplexity
                continue
            else:
                k -= steps
                steps = steps//2
        # end
        return k

    def _set_perplexity(self, k, perplexity):
        if perplexity == 0:
            self._perplexity = [0. for i in range(k)]
        else:
            self._perplexity[k] = perplexity
        return perplexity

    def _train_model(self, corpus: list[list[str]], k: int):
        self._log.info(f"... train k={k}")
        n = len(corpus)
        docids = []

        mdl = tp.LDAModel(k=k)

        for i in range(n):
            doc = corpus[i]
            docid = mdl.add_doc(doc)
            if docid == None:
                self._log.warn(f"Unable to process document {i}: len={len(doc)}")
                continue
            else:
                docids.append(docid)
        # end

        min_perplexity = 10000
        p = 0
        it = 0
        while p <= self.patience:
            it += 1
            mdl.train(10)
            perplexity = mdl.perplexity
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                p = 1
                self._log.debug(f"... ... [{it:3}/{p:2}] perplexity: {perplexity:.5}")
            else:
                p += 1
                self._log.debug(f"... ... [{it:3}/{p:2}] perplexity: {perplexity:.5}")

        self._log.info(f"... [{k}] perplexity: {min_perplexity:.5}")

        self._docids = docids
        self._mdl = mdl

        return mdl
    # end

    def _compute_distributions(self):
        mdl = self._mdl

        # tokens
        self._tokens = list(mdl.used_vocabs)

        # tokens distribution in topics
        tdist = []
        for i in range(self.k):
            wdist = mdl.get_topic_word_dist(i, True)
            tdist.append(wdist)
        self._topics = np.array(tdist)

        # topics distribution in documents
        ddist = []
        for docid in self._docids:
            doc = mdl.docs[docid]
            res = mdl.infer(doc)
            ddist.append(res[0])
        self._documents = np.array(ddist)

        # topic names
        self._topic_names = topic_names(self.tokens, self.topics)
    # end

    # -----------------------------------------------------------------------
    # IO
    # -----------------------------------------------------------------------

    def save(self, file: str):
        """

        :param path: string containing '%s'.
            It will be replaced with 'topics' and 'documens'
        :return: None
        """
        self.save_ordered_topics(file)
        self.save_ordered_documents(file)
        self.save_topics(file)
        self.save_documents(file)
    # end

    def save_topics(self, file):
        file = file.replace('%s', 'topics')
        self._log.info(f"saving '{file}'")

        tokens = self.tokens
        topic_names = self.topic_names
        tokens_matrix = self.topics
        n_topics = len(topic_names)

        with open(file, mode='w') as wrt:
            head = ",".join(['topicname'] + tokens)
            wrt.write(head + "\n")

            for i in range(n_topics):
                topic_name = topic_names[i]
                record = ",".join([topic_name] + list(map(str, tokens_matrix[i])))
                wrt.write(record + "\n")
            # end
        # end
        pass
    # end

    def save_documents(self, file):
        file = file.replace('%s', 'documents')
        self._log.info(f"saving '{file}'")

        topic_names = self.topic_names
        file_names = self._file_names
        doc_matrix = self.documents
        n_files = len(file_names)

        with open(file, mode='w') as wrt:
            head = ",".join(['filename'] + topic_names)
            wrt.write(head + "\n")

            for i in range(n_files):
                file_name = file_names[i]
                record = ",".join([file_name] + list(map(str, doc_matrix[i])))
                wrt.write(record + "\n")
            # end
        # end
        pass
    # end

    def save_ordered_topics(self, file):
        file = file.replace('%s', 'ordered-topics')
        self._log.info(f"saving '{file}'")

        topic_names = self.topic_names
        tokens = self.tokens
        tokens_matrix = self.topics
        n_topics = len(topic_names)

        tokens_distrib = LabelsDistribution(labels=tokens, distributions=tokens_matrix)

        with open(file, mode='w') as wrt:
            for i in range(n_topics):
                record = [topic_names[i]]
                tok_distrib = tokens_distrib.sorted_distrib(i)
                for t, w in tok_distrib:
                    record += [t, str(w)]

                wrt.write(",".join(record) + "\n")
        # end
        pass
    # end

    def save_ordered_documents(self, file):
        file = file.replace('%s', 'ordered-documents')
        self._log.info(f"saving '{file}'")

        file_names = self._file_names
        topic_names = self.topic_names
        topics_matrix = self.documents
        n_documents = len(file_names)

        topics_distrib = LabelsDistribution(labels=topic_names, distributions=topics_matrix)

        with open(file, mode='w') as wrt:
            for i in range(n_documents):
                record = [file_names[i]]
                top_distrib = topics_distrib.sorted_distrib(i)
                for t, w in top_distrib:
                    record += [t, str(w)]

                wrt.write(",".join(record) + "\n")
        # end
        pass
    # end

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
