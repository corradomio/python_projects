import requests
from typing import Tuple, Optional


class ArcadeClient:
    def __init__(self, url:str, database: str,  auth: Tuple[str,str]):
        self.url = url
        self.database = database
        self.auth = auth

        self.api_url = f"{self.url}/api/v1/command/{self.database}"
        self.session_id: Optional[str] = None
    # end

    def create_database(self, name: str):
        resp = requests.post(
            url=f"{self.url}/api/v1/create/{self.database}",
            json=dict(),
            auth=self.auth,
            headers={"Content-Type": "application/json"})
        return resp.json()
    # end

    def beginTransaction(self):
        resp = requests.post(
            url=f"{self.url}/api/v1/begin/{self.database}",
            json=dict(),
            auth=self.auth,
            headers={"Content-Type": "application/json"})
        jresp = resp.json()
        self.session_id = jresp["arcadedb-session-id"]
        return jresp
    # end

    def commit(self):
        if self.session_id is None:
            raise Exception("Session not started")

        resp = requests.post(
            url=f"{self.url}/api/v1/commit/{self.database}",
            json=dict(),
            auth=self.auth,
            headers={"Content-Type": "application/json", "arcadedb-session-id": self.session_id})
        jresp = resp.json()
        self.session_id = jresp["arcadedb-session-id"]
        return jresp
    # end

    def rollback(self):
        if self.session_id is None:
            raise Exception("Session not started")

        resp = requests.post(
            url=f"{self.url}/api/v1/commit/{self.database}",
            json=dict(),
            auth=self.auth,
            headers={"Content-Type": "application/json", "arcadedb-session-id": self.session_id})
        jresp = resp.json()
        self.session_id = jresp["arcadedb-session-id"]
        return jresp
    # end

    def run(self, s, **kwargs):
        hdrs = {"Content-Type": "application/json"}
        if self.session_id is not None:
            hdrs["arcadedb-session-id"] = self.session_id

        resp = requests.post(
            url=self.api_url,
            json={
                "language": "cypher",
                "command": s,
                "params": kwargs,
            },
            auth=self.auth,
            headers=hdrs)

        return resp.json()
    # end
# end


class GraphDatabase:

    @staticmethod
    def driver(url: str, auth: Tuple[str, str]) -> ArcadeClient:
        # http://server:port/database
        p = url.rfind('/')

        return ArcadeClient(url=url[0:p], database=url[p+1:], auth=auth)
    # end
# end
