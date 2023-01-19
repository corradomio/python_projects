from pprint import pprint
from arcadecli import GraphDatabase


# api_url = "http://172.23.86.3:2480/api/v1/command/Test"
#
# body = {
#     "language": "cypher",
#     "command": "create (n:Test {name:$name}) return n",
#     "params": {
#         "name": "Pinco"
#     }
# }

# pprint(json.dumps(body))

# response = requests.post(api_url,
#                          json=body,
#                          auth=('root', 'playwithdata'),
#                          # data=json.dumps(body),
#                          headers={"Content-Type": "application/json"})
#
# pprint(response.json())

acli = GraphDatabase.driver("http://172.23.86.3:2480/Test", auth=('root', 'playwithdata'))

# acli = ArcadeClient(url="http://172.23.86.3:2480", database="Test", auth=('root', 'playwithdata'))

pprint(acli.run("match (n:Test) return count(n)"))


