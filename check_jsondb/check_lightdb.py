#
# https://github.com/Fl1yd/LightDB
#

from typing import List, Dict, Any

from lightdb import LightDB
from lightdb.models import Model

db = LightDB("lightdb.json")

class User(Model, table="users"):
    name: str
    age: int
    items: List[str] = []
    extra: Dict[str, Any] = {}

user = User.create(name="Alice", age=30)

user = User.get(User.name == "Alice")
# or user = User.get(name="Alice")
print(user.name, user.age)


user.name = "Kristy"
user.save()

users = User.filter(User.age >= 20)
for user in users:
    print(user.name)


