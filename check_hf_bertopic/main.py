from pprint import pprint
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

print("fetch_20newsgroups")
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

print("BERTopic")
topic_model = BERTopic()
print("fit_transform")
topics, probs = topic_model.fit_transform(docs)

pprint(topic_model.get_topic_info())

pprint(topic_model.get_topic(0))

