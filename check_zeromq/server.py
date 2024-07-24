import time
import zmq
import stdlib.logging as logging


def main():
    log = logging.getLogger("main")
    log.info("Start server")

    context = zmq.Context()

    # server->REP, client->REQ
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    # zmqs = ZMQEventHandler(socket)

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print(f"Received request: {message}")

        #  Do some 'work'
        time.sleep(1)

        #  Send reply back to client
        socket.send_string("World")

    # time.sleep(3600)

    log.info("End")
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
