from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
refId = '4475dd0c'


def get_sources(tx, refId):
    sources = []
    result = tx.run("MATCH (n:source {refId:$refId}) RETURN n", refId=refId)
    for record in result:
        sources.append(record["n"])
    return sources


def get_source_links(tx, refId):
    links = []
    result = tx.run("MATCH (n:source {refId:$refId})-->(m:source {refId:$refId}) RETURN id(n) AS s, id(m) AS t",
                    refId=refId)
    for record in result:
        links.append( (record["s"], record["t"]) )
    return links


with driver.session() as s:
    links = s.execute_read(get_source_links, refId)
    with open("../acme_source_edges.csv", mode="w") as f:
        f.write("source,target\n")
        for link in links:
            print(link)
            f.write(f"{link[0]},{link[1]}\n")

driver.close()
