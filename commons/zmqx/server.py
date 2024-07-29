import time
from typing import Any
import zmq

from .base import ZMQBase, MessageType


class ZMQServer(ZMQBase):
    def __init__(self, stype: int, url: str, *, mtype: MessageType = MessageType.STRING,
                 timeout=5, name=None, daemon=None):
        """

        :param type: socket type
        :param url: url to use for the bind
        :param timeout: timeout to receive messages (in seconds)
        :param name: thread name
        :param mtype: message type
            None: as is
            'json': JSON
            'pickle': serialized using pickle
        :param daemon: ???
        """
        super().__init__(stype=stype, url=url,
                         mtype=mtype, timeout=timeout,
                         name=name, daemon=daemon)

    def run(self):
        try:
            context = zmq.Context()
            socket = context.socket(self._stype)
            socket.set(zmq.RCVTIMEO, -1 if self._timeout < 0 else 1000*self._timeout)
            socket.bind(self._url)

            self._socket = socket

            while not self._interrupted:
                message = self.recv()

                if message is None:
                    continue

                try:
                    response = self.on_message(message)
                except Exception as e:
                    response = None

                self.send(response)
            # end
        except Exception as e:
            pass
    # end

    def on_message(self, message) -> Any:
        print(f"Received request: {message}")

        time.sleep(1)

        return "World"

    def interrupt(self):
        """Terminate the server"""
        self._interrupted = True

# end
