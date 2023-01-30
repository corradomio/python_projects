from gensim.corpora import Dictionary
from gensim.models import LdaModel

from common import *
from topics import GensimTopics

def nameof(file):
    # remove the path and '.java' extension
    file = file.replace('\\', '/')
    p = file.rindex('/')
    e = file.rindex('.')
    return file[p+1:e]

def save_topics(files, corpus, model):
    top_topics = model.top_topics(corpus)

    topics_file = f"{NAME}-topic.csv"
    with open(topics_file, mode="w", encoding="UTF-8") as f:
        for topic in top_topics:
            # tdict = [(p[1], p[0]) for p in topic[0]]
            tdict = [p[1] for p in topic[0]]
            f.write(f"{','.join(tdict)}\n")

    doc_topic_file = f"{NAME}-doc-topic.txt"
    with open(doc_topic_file, mode="w", encoding="UTF-8") as f:
        n = len(files)
        for i in range(n):
            file = files[i]
            name = nameof(file)
            doc = corpus[i]
            doc_topics = model.get_document_topics(doc)
            f.write(f"{name}, {doc_topics}\n")
    # end
# end


def main():
    docs, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS)

    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

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

    save_topics(files, corpus, model)

    # gst = GensimTopics(dictionary, top_topics)
    # # Average topic coherence is the sum of topic coherence of all topics, divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    # print('Average topic coherence: %.4f.' % avg_topic_coherence)
    #
    # from pprint import pprint
    # pprint(top_topics)

    pass
# end


if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
