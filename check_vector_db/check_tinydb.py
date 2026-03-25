#
# https://tinydb.readthedocs.io/en/latest/
#

from tinydb import TinyDB, Query
db = TinyDB('tinydb.json')
User = Query()
db.insert({'name': 'John', 'age': 22})
print(db.search(User.name == 'John'))

