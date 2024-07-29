import zmq
from .base import ZMQBase, MessageType


class ZMQClient(ZMQBase):

    def __init__(self, stype: int, url: str, *, mtype: MessageType = MessageType.STRING, timeout=5):
        super().__init__(stype=stype, url=url,
                         mtype=mtype, timeout=timeout)

        context = zmq.Context()
        socket = context.socket(stype)
        socket.set(zmq.RCVTIMEO, -1 if self._timeout < 0 else 1000*self._timeout)
        socket.connect(self._url)

        self._socket = socket

    def send_receive(self, message):
        self.send(message)
        return self.recv()

# end
