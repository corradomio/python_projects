from pprint import pprint
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

LINUX = "192.168.0.123"
WINDOWS = "192.168.161.139"

bc = BertClient(ip=LINUX)
# bc = BertClient(ip=WINDOWS)
encoded = bc.encode(['First do it', 'then do it right', 'then do it better'])
""":type: Union[Type[np.ndarray]]"""

pprint(encoded.shape)
pprint(encoded)


king = bc.encode(['king'])
qeen = bc.encode(['qeen'])
male = bc.encode(['male'])
female = bc.encode(['female'])

whois = king - male + female

pprint(["king->male",    cosine_similarity(king, male)[0][0]])
pprint(["qeen->female",  cosine_similarity(qeen, female)[0][0]])
pprint(["qeen->whois",   cosine_similarity(qeen, whois)[0][0]])
pprint(["whois->female", cosine_similarity(whois, female)[0][0]])

girl  = bc.encode(['girl'])
boy   = bc.encode(['boy'])
young = bc.encode(['young'])
old   = bc.encode(['old'])
woman = bc.encode(['woman'])
man   = bc.encode(['man'])

pprint(["girl-young+old->woman", cosine_similarity(woman, girl - young + old)[0][0]])
pprint(["girl-young+old->man",   cosine_similarity(man,   girl - young + old)[0][0]])
pprint(["boy-young+old->man",    cosine_similarity(man,   boy  - young + old)[0][0]])



