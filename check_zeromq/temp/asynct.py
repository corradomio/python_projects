from tornado import gen, ioloop
import zmq
from zmq.eventloop.future import Context
from zmq.eventloop.zmqstream import ZMQStream

ctx = Context.instance()

@gen.coroutine
def recv():
    s = ctx.socket(zmq.SUB)
    s.connect('tcp://127.0.0.1:5555')
    s.subscribe(b'')
    while True:
        msg = yield s.recv_multipart()
        print('received', msg)
    s.close()


s = ctx.socket(zmq.REP, socket_class=zmq.Socket)
s.bind('tcp://localhost:5555')
stream = ZMQStream(s)

def echo(msg):
    stream.send_multipart(msg)

stream.on_recv(echo)
ioloop.IOLoop.instance().start()


print("done")
