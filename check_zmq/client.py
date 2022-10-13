#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import jzmq

context = zmq.Context()

#  Socket to talk to server
# print("Connecting to hello world server…")
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://localhost:5555")
#
# for request in range(10):
#     print("Sending request %s …" % request)
#     socket.send(b"Hello")
#
#     #  Get the reply.
#     message = socket.recv()
#     print("Received reply %s [ %s ]" % (request, message))

socket = jzmq.JZMQ("tcp://localhost:5555")
socket.start_client(zmq.REQ)

#  Do 10 requests, waiting each time for a response
for request in range(10):
    message = {1: "Hello", 2: str(request)}
    print("Sending  %s …" % message)
    socket.send(message)

    #  Get the reply.
    response = socket.recv()
    print("Received %s " % response)
