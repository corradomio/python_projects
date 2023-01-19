from neo4j import GraphDatabase

#

# uri = "neo4j://localhost:7687"
uri = "neo4j://102.37.140.82:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# "127ef6fc"	"Flink-test"
# "4a95c5c1"	"TomCat"
# "f92ff479"	"Camel"
# "1bccefb2"	"Hibernate-orm"
# "e2d0d731"	"Hibernate-reactive"
# "11aa4e33"	"hibernate-searchs"

# "4a95c5c1"	4111
# "127ef6fc"	21015
# "1bccefb2"	17152
# "e2d0d731"	789
# "11aa4e33"	7377
# "f92ff479"	27792

PROJECTS = {
    # '127ef6fc': 'flink',
    '4a95c5c1': 'tomcat',
    'f92ff479': 'camel',
    '1bccefb2': 'hibernate-orm',
    '11aa4e33': 'hibernate-search'
}


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


def dump_sources(refId, name):
    print(f"Dump sources {name}/{refId}")
    with driver.session() as s:
        links = s.execute_read(get_source_links, refId)
        with open(f"{name}_source_edges.csv", mode="w") as f:
            f.write("source,target\n")
            for link in links:
                print(link)
                f.write(f"{link[0]},{link[1]}\n")

    driver.close()



# def get_sources(tx, refId):
#     sources = []
#     result = tx.run("MATCH (n:source {refId:$refId}) RETURN n", refId=refId)
#     for record in result:
#         sources.append(record["n"])
#     return sources


def get_type_links(tx, refId):
    links = []
    result = tx.run("MATCH (n:type {refId:$refId})-->(m:type {refId:$refId}) "
                    "WHERE n.type <> 'reftype' "
                    "  AND m.type <> 'reftype' "
                    "RETURN id(n) AS s, id(m) AS t",
                    refId=refId)
    for record in result:
        links.append( (record["s"], record["t"]) )
    return links


def dump_types(refId, name):
    print(f"Dump types {name}/{refId}")
    with driver.session() as s:
        links = s.execute_read(get_type_links, refId)
        with open(f"{name}_type_edges.csv", mode="w") as f:
            f.write("source,target\n")
            for link in links:
                print(link)
                f.write(f"{link[0]},{link[1]}\n")

    driver.close()


def main():
    for refId in PROJECTS:
        name = PROJECTS[refId]

        dump_types(refId, name)
        dump_sources(refId, name)


if __name__ == "__main__":
    main()
