import zmq
import json
import pickle


class JZMQ:

    def __init__(self, url: str):
        self._url: str = url
        self.socket = None
        self.context = None
    # end

    def start_server(self, proto: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(proto)
        self.socket.bind(self._url)

        while True:
            data = self.socket.recv()
            message = self._frombytes(data)
            if message == 'stop':
                break
            response = self._dispatch(message)
            data = self._tobytes(response)
            self.socket.send(data)
    # end

    def _tobytes(self, message):
        # smsg: str = json.dumps(message)
        # data = smsg.encode('utf-8')
        data = pickle.dumps(message)
        return data

    def _frombytes(self, data):
        # smsg = data.decode('utf-8')
        # message = json.loads(smsg)
        message = pickle.loads
        return message

    def _dispatch(self, message):
        print(message)
        return message

    def start_client(self, proto: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(proto)
        self.socket.connect(self._url)
    # end

    def send(self, message):
        data = self._tobytes(message)
        # data = message
        self.socket.send(data)
    # end

    def recv(self):
        data = self.socket.recv()
        # message = data
        message = self._frombytes(data)
        return message
    # end
# end