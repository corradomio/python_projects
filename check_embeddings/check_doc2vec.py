from common import *
import gensim


def gen_embeddings(vsz):
    corpus, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS)
    fileids = load_fileids()
    n = len(corpus)

    train_corpus = [gensim.models.doc2vec.TaggedDocument(corpus[i], [i]) for i in range(n)]

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vsz, min_count=2, epochs=100)
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    fname = f"{PROJECT}-doc2vec-{vsz}.vec"
    with open(fname, mode='w') as wrt:
        for i in range(n):
            file = files[i]
            if file not in fileids:
                print(f"file skipped: '{file}'")
                continue

            doc = corpus[i]
            dvect = model.infer_vector(doc)

            svect = list(map(str, dvect))
            semb = ",".join(svect)

            fileid = fileids[file]
            wrt.write(f"{fileid},{semb}\n")
    # end
    pass


def main():
    gen_embeddings(50)
    gen_embeddings(10)
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

