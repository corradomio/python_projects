import threading
import typing
import zmq
from enum import IntEnum


class MessageType(IntEnum):
    NATIVE = 0,
    STRING = 1,
    JSON = 2,
    PYOBJ = 3,
    PICKLE = 3


class ZMQBase(threading.Thread):
    def __init__(self,
                 stype: int, url: str, *,
                 mtype: MessageType = MessageType.NATIVE, timeout=5,
                 name=None, daemon=None):
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
        threading.Thread.__init__(self, name=name, daemon=daemon)
        self._stype = stype
        self._url = url
        self._interrupted = False
        self._timeout = timeout
        self._mtype = mtype
        self._socket = None
    # end

    def send(self, message):
        socket = self._socket
        mtype = self._mtype

        try:
            if message is None:
                pass
            elif mtype == MessageType.JSON:
                socket.send_json(message)
            elif mtype == MessageType.PYOBJ:
                socket.send_pyobj(message)
            elif mtype == MessageType.STRING:
                socket.send_string(message)
            else:
                socket.send(message)
        except Exception as e:
            pass
    # end

    def recv(self) -> typing.Any:
        socket = self._socket
        mtype = self._mtype

        try:
            if mtype == MessageType.JSON:
                message = socket.recv_json()
            elif mtype == MessageType.PYOBJ:
                message = socket.recv_pyobj()
            elif mtype == MessageType.STRING:
                message = socket.recv_string()
            else:
                message = socket.recv()
        except zmq.Again:
            message = None

        return message
    # end
# end
