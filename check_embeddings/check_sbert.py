from sentence_transformers import SentenceTransformer
from common import *


def save_embeddings():
    corpus, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS)
    n = len(files)
    fileids = load_fileids()

    corpus = list(map(lambda t: " ".join(t), corpus))

    model = SentenceTransformer('all-MiniLM-L6-v2')
    dvects = model.encode(corpus, normalize_embeddings=True)

    fname = f"cocome-bert.vec"
    with open(fname, mode='w') as wrt:
        for i in range(n):
            file = files[i]
            if file not in fileids:
                print(f"file skipped: '{file}'")
                continue

            # doc = corpus[i]
            # dvect = model.encode(doc, normalize_embeddings=True)
            dvect = dvects[i]

            svect = list(map(str, dvect))
            semb = ",".join(svect)

            fileid = fileids[file]
            wrt.write(f"{fileid},{semb}\n")
    pass
def main():
    save_embeddings()
# end


if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
