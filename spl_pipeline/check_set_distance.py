import warnings
import logging.config
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import sklearn.cluster as sklc

from spl import ProjectFiles, FilesTokenizer, LDATopicModel, LabelsDistribution
from spl.utils_numpy import pairwise_distances
from spl.utils_stdlib import flatten
from spl.corpus_distances import jaccard_distance
import matplotlib.pyplot as plt


def main():
    log.info("main")

    pl = ProjectFiles()

    # pl_name = 'cocome-%s.csv'
    # pl.scan(r'D:\SPLGroup\spl-workspaces\java\cocome-maven-project')

    # pl_name = 'dl4j-%s.csv'
    # pl.scan(r'D:\Projects.github\other_projects\deeplearning4j-1.0.0-M2')

    pl_name = 'elasticsearch-%s.csv'
    pl.scan(r'D:\Projects\Java\elasticsearch-8.1.2')

    ft = FilesTokenizer(stem=True,
                        min_len=3,
                        min_count=3,
                        min_tfidf=0.001,
                        unique=True,
                        stopwords='java')
    ft.fit(pl.files)

    corpus = ft.corpus
    jdmap = jaccard_distance(corpus)
    jdvec = jdmap.reshape(-1)

    plt.matshow(jdmap)
    plt.show()

    plt.hist(jdvec, bins=50)
    plt.show()

    pass




if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
