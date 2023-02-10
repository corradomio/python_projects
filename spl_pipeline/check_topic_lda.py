import warnings
import logging.config
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import sklearn.cluster as sklc

from spl import ProjectFiles, FilesTokenizer, LDATopicModel, LabelsDistribution
from spl.utils_numpy import pairwise_distances
from spl.utils_stdlib import flatten

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
log = None


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
    ft.save(pl_name)

    tm = LDATopicModel(k=0)
    tm.fit(ft.corpus, names=ft.names)
    tm.save(pl_name)

    T = tm.topics
    X = tm.documents

    print(T.shape)
    print(X.shape)

    # tdist = LabelsDistribution(tm.tokens, tm.topics)
    # labels = tdist.sorted_distributions(.5, label_only=True)
    # pprint(labels)
    # print(len(set(flatten(labels))))

    # ddist = LabelsDistribution(tm.topic_names, tm.documents)
    # labels = ddist.sorted_distributions(.5, label_only=True)
    # pprint(labels)
    # print(len(set(flatten(labels))))

    # distances
    # 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    # 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    # 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    # 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    # 'yule'
    metric = 'cosine'
    docdist = pairwise_distances(X, metric=metric)
    print(f"min: {docdist.min()}, max:{docdist.max()}, mean:{docdist.mean()}")
    plt.matshow(docdist)
    plt.show()

    # for eps in [0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #     dbscan = sklc.DBSCAN(eps=eps, metric=metric, n_jobs=4)
    #     dbscan.fit(X)
    #     labels = dbscan.labels_
    #     print(f"eps: {eps}:", len(set(labels)), ":", Counter(labels))
    for k in [3, 6, 10, 20, 40, 70]:
        km = sklc.KMeans(n_clusters=k)
        km.fit(X)
        labels = km.labels_
        print(f"k: {k}:", len(set(labels)), ":", Counter(labels))

    # dbscan = sklc.DBSCAN(eps=0.05, metric=metric, n_jobs=4)
    # dbscan.fit(X)
    # labels = dbscan.labels_
    # n_labels = max(labels) + 2  # +1 for label '-1' and +1 because the first label has index '0'
    # print(Counter(labels))

    # print(tm.tokens)
    # print(tm.topics.shape)
    # for tn in tm.topic_names:
    #     print(tn)
    # print(tm.documents.shape)
    pass



if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
