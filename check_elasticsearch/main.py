import os
import json
from elasticsearch import Elasticsearch

elasticsearch = {
    "hosts": ["https://10.193.20.84:50051"],
    "user": "python_writer",
    "password": "python_writer@19791007",
    "index": "prams_2026",
    "ca_certs": "ca_1717136426129.crt",
    "_id_fields": ["participant_id"],
    "number_of_records": 10000
}

es_hosts = elasticsearch["hosts"]
es_user = elasticsearch["user"]
es_pass = elasticsearch["password"]
es_index = elasticsearch["index"]
es_size = elasticsearch["number_of_records"]

root_path = r"D:/Projects.ebtic/project.diwang/prams/"
es_ca_certs = elasticsearch["ca_certs"]

def main():
    es_connection = Elasticsearch(
        es_hosts,
        verify_certs=True,
        basic_auth=(es_user, es_pass),
        ca_certs=root_path + es_ca_certs
    )

    resp = es_connection.search(
        index=es_index,
        size=es_size
    )
    body = resp.body

    hits = body["hits"]["hits"]

    for h in hits:
        _id = h["_id"]
        d = h["_source"]

        participant_name = d.get("participant_name", "")
        if participant_name is None:
            participant_name = ""
        participant_contact_nb = d.get("participant_contact_nb", "")
        if participant_contact_nb is None:
            participant_contact_nb = ""

        if len(participant_name) == 0 and len(participant_contact_nb):
            continue

        if "participant_name" in d:
            del d["participant_name"]
        if "participant_contact_nb" in d:
            del d["participant_contact_nb"]

        es_connection.index(index=es_index, id=_id, document=d)
        print("Updated", _id)
        pass
    print(resp)


if __name__ == "__main__":
    main()
