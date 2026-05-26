from pathlib import Path
import os
from stdlib import jsonx
from elasticsearch import Elasticsearch

elasticsearch = {
    "hosts": ["https://10.193.20.11:50051/"],
    "user": "lab_monitoring_developer",
    "password": "lab123",
    "index": "lab_monitor_version_14nov_summary",
    "ca_certs": "develk-10.pem",
}

es_hosts = elasticsearch["hosts"]
es_user = elasticsearch["user"]
es_pass = elasticsearch["password"]
es_index = elasticsearch["index"]

root_path = r"."
es_ca_certs = elasticsearch["ca_certs"]


def name_of(f: Path):
    name = f.name
    pos = name.rfind(".")
    return name[:pos]

def main():
    es = Elasticsearch(
        es_hosts,
        verify_certs=False,
        basic_auth=(es_user, es_pass),
        ca_certs=root_path + "/" + es_ca_certs
    )

    # es_mappings={"mappings":{}}
    # es.indices.create(index=es_index, body=es_mappings)

    ROOT = Path(".")

    for f in ROOT.glob("*.json"):
        data = jsonx.load(f)

        safe_id = name_of(f)

        es.index(index=es_index, id=safe_id, document=data)
        pass



    # resp = es_connection.search(
    #     index=es_index,
    #     size=es_size
    # )
    # body = resp.body
    #
    # hits = body["hits"]["hits"]
    #
    # for h in hits:
    #     _id = h["_id"]
    #     d = h["_source"]
    #
    #     participant_name = d.get("participant_name", "")
    #     if participant_name is None:
    #         participant_name = ""
    #     participant_contact_nb = d.get("participant_contact_nb", "")
    #     if participant_contact_nb is None:
    #         participant_contact_nb = ""
    #
    #     if len(participant_name) == 0 and len(participant_contact_nb):
    #         continue
    #
    #     if "participant_name" in d:
    #         del d["participant_name"]
    #     if "participant_contact_nb" in d:
    #         del d["participant_contact_nb"]
    #
    #     es_connection.index(index=es_index, id=_id, document=d)
    #     print("Updated", _id)
    #     pass
    # print(resp)


if __name__ == "__main__":
    main()
