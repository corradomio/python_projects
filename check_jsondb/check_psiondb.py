#
# https://github.com/pysonDB/pysonDB-v2
#

from pysondb import PysonDB

db = PysonDB('psiondb.json')

jid = db.add({'name': 'John', 'age': 22})
print(jid)
print(db.get_by_query(lambda d: d['age'] <= 30))
