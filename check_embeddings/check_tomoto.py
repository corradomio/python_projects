from common import *

def save_embeddings(lda_k):
    corpus, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS)
    fileids = load_fileids()
    docids = []
    n = len(files)

    lda_k = 10

    mdl = tp.LDAModel(k=lda_k, min_cf=4, min_df=4)
    for doc in corpus:
        docid = mdl.add_doc(doc)
        docids.append(docid)

    for i in range(0, 100, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    print(mdl.summary())

    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))

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
    save_embeddings(10)
    save_embeddings(20)
# end


if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
