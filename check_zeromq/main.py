from typing import Any

import zmqx
import time
from stdlib.tprint import tprint


class MainServer(zmqx.ZMQServer):
    def __init__(self, stype: int, url: str, *, mtype=None, timeout=5, name=None, daemon=None):
        super().__init__(stype=stype, url=url, mtype=mtype, timeout=timeout, name=name, daemon=daemon)

    def on_message(self, message) -> Any:
        print(f"Received request: {message}")
        return {
            "resp": "World"
        }

    # def on_message(self, socket, message):
    #     print(f"Received request: {message}")
    #
    #     time.sleep(1)
    #
    #     #  Send reply back to client
    #     socket.send_string("World")


def main():
    tprint("create 0mq server")
    zs = zmqx.ZMQServer(zmqx.REP, "tcp://localhost:5555", mtype=zmqx.MessageType.JSON)

    tprint("start 0mq server")
    zs.start()
    # zs.run()

    tprint("main sleep")
    time.sleep(3600)

    tprint("main done")
# end


if __name__ == "__main__":
    main()
