from common import *

def save_embeddings(lda_k):
    corpus, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS)
    fileids = load_fileids()
    docids = []
    n = len(files)

    mdl = tp.LDAModel(k=lda_k, min_cf=4, min_df=4)
    for doc in corpus:
        docid = mdl.add_doc(doc)
        docids.append(docid)

    for i in range(0, 100, 10):
        mdl.train(10)
        # print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    # print(mdl.summary())
    print(f"[{lda_k}] Perplexity: {mdl.perplexity}")

    for k in range(mdl.k):
        # print('Top 10 words of topic #{}'.format(k))
        # print(mdl.get_topic_words(k, top_n=10))
        words = mdl.get_topic_words(k, top_n=10)
        words = list(map(lambda t: t[0], words))
        words = ",".join(words)
        print(f"{k},{words}")
        continue

    # save the words in a file
    fname = f"cocome-tomoto-topic-{lda_k}.vec"
    with open(fname, mode='w') as wrt:
        wrt.write(f"{','.join(mdl.used_vocabs)}\n")

        for k in range(mdl.k):
            wdist = mdl.get_topic_word_dist(k)
            wdist = list(map(str, wdist))
            wrt.write(f"{','.join(wdist)}\n")


    fname = f"cocome-tomoto-{lda_k}.vec"
    with open(fname, mode='w') as wrt:
        for i in range(n):
            file = files[i]
            if file not in fileids:
                print(f"file skipped: '{file}'")
                continue

            docid = docids[i]
            dinst = mdl.docs[docid]
            dvect = mdl.infer(dinst)[0]

            svect = list(map(str, dvect))
            semb = ",".join(svect)

            fileid = fileids[file]
            wrt.write(f"{fileid},{semb}\n")
    pass


def main():
    save_embeddings( 10)
    save_embeddings( 20)
    save_embeddings( 50)
    save_embeddings(100)
    save_embeddings(150)
    save_embeddings(300)
    save_embeddings(500)
    save_embeddings(700)
# end


if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
