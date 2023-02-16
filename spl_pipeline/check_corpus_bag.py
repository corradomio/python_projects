import logging.config
from spl import ProjectFiles, FilesTokenizer, LDATopicModel, LabelsDistribution
from spl.utils_numpy import pairwise_distances
from spl.utils_stdlib import flatten


def main():
    log.info("main")

    pl = ProjectFiles()

    pl_name = 'cocome-%1s-%s.csv'
    pl.scan(r'D:\Projects\Java\\cocome-maven-project')

    # pl_name = 'dl4j-%1s-%s.csv'
    # pl.scan(r'D:\Projects\Java\\deeplearning4j-1.0.0-M2')

    # pl_name = 'elasticsearch-%1s-%s.csv'
    # pl.scan(r'D:\Projects\Java\elasticsearch-8.1.2')

    ft = FilesTokenizer(stem=True,
                        min_len=3,
                        min_count=3,
                        min_tfidf=0.001,
                        # min_len=0,
                        # min_count=0,
                        # min_tfidf=0,

                        unique=False,
                        stopwords='java')
    ft.fit(pl.files)

    corpus = ft.get_corpus(bag=True, normalized=True)
    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
