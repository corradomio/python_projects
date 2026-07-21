import json
import warnings

from urllib3.exceptions import InsecureRequestWarning, SecurityWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
warnings.filterwarnings("ignore", category=SecurityWarning)

from elasticsearch import Elasticsearch
from stdlib.dictx import dict_get as dget

def main():
    kibana_url = "http://10.193.20.11:50054/"

    # server_id = "https://127.0.0.1:9200/"
    server_id = "https://10.193.20.11:50051/"
    user_name = "lab_monitoring_developer"
    elastic_pass = "lab123"

    index_name = "lab_monitor_version_13jul"

    es = Elasticsearch(
        [server_id],
        basic_auth=(user_name, elastic_pass),
        verify_certs=False,
        request_timeout=10
    )

    # print(es.indices.exists(index=index_name))

    # ret = es.search(index=index_name, body={"query": {"match_all": {}}})
    # ret = es.search(index=index_name, body={"query": {"range": {
    #     "present_start":{
    #         "gte": "2026-06-24"
    #     }
    # }}})
    # print(ret)

    # ret = es.search(index=index_name, body={"query": {
    #     "bool":{
    #         "must": [
    #             # {"range":{
    #             #     "present_start":{"gte": "2026-01-21T00:00:00+00:00"}
    #             # }}
    #             # ,
    #             # {"range":{
    #             #     "present_end":{"lte": "2026-01-21T23:59:59+00:00"}
    #             # }}
    #             # ,
    #             {"term": {
    #                 "person_name" : {
    #                     "value": "P@260121_001"
    #                 }
    #             }}
    #         ]
    #         # ,
    #         # "filter": [
    #         #     {"match": { "person_name": "P@260121_001" } }
    #         # ]
    #     }
    # }})

    q = {"query": {
        "bool": {
            "must": {
                "match_all": {}
            },
            "filter": {
                "term": {
                    "status": "active"
                }
            }
        }
    }}

    q = {"query": {
        "bool":{
            "filter": [
                {"term": {
                    "person_name" : {
                        "value": "P@260121_001"
                    }
                }}
            ]
        }
    }}

    q = {"query": {
        "match": {
            "person_name": {
                "query": "P@260121_001"
            }
        }

    }}

    day = "2026-06-30"

    q = {"query": {
        "bool": {
            "must": [

                # {
                #     "term": {"person_name.keyword": "P@260121_001"}
                # }
                # ,
                # {
                #     "term": {"track_status.keyword": "combined"}
                # }

                {
                    "range": {
                        "present_start": {
                            "gte": day + "T00:00:00+04:00",
                            "lte": day + "T23:59:59+04:00",
                        }
                    }
                }
            ]
        }

    }}

    ret = es.search(index=index_name, body=q)


    print(dget(ret.body, "hits.total.value"))

    # for r in dget(ret.body, "hits.hits"):
    #     print(dget(r, "_id"), ":", json.dumps(dget(r, "_source"), indent=2))

    # print(json.dumps(ret.body, ))

    pass



if __name__ == "__main__":
    main()

