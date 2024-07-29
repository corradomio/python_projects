import threading
import zmq
import time


class ServerResponse(threading.Thread):

    def __init__(self):
        threading.Thread.__init__ (self)

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")

        while True:
            #  Wait for next request from client
            message = socket.recv()
            print(f"Received request: {message}")

            #  Do some 'work'
            time.sleep(1)

            #  Send reply back to client
            socket.send_string("World")
        # end
    # end
# end



def main():

    sr = ServerResponse()

    print("server start")
    sr.start()

    print("server sleep")
    time.sleep(3600)

    print("server join")
    sr.join()

    print("server end")
    pass


if __name__ == "__main__":
    main()