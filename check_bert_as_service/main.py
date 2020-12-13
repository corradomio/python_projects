from pprint import pprint
from bert_serving.client import BertClient
bc = BertClient(ip="192.168.0.123")
encoded = bc.encode(['First do it', 'then do it right', 'then do it better'])

pprint(encoded)
