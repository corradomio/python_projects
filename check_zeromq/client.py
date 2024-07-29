import stdlib.logging as logging
import zmq
import zmqx
import time


def main():
    log = logging.getLogger("main")
    log.info("Start client")

    cli = zmqx.ZMQClient(zmq.REQ, "tcp://localhost:5555", mtype=zmqx.MessageType.JSON)

    for request in range(10):
        print(f"Sending request {request} ...")

        message = cli.send_receive({
            "req": "Hello"}
        )

        print(f"Received reply {request} [ {message} ]")

    log.info("End")
    pass


def main1():
    log = logging.getLogger("main")
    log.info("Start client")

    context = zmq.Context()

    # server->REP, client->REQ
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    for request in range(10):
        print(f"Sending request {request} ...")
        socket.send_string("Hello")

        #  Get the reply.
        message = socket.recv()
        print(f"Received reply {request} [ {message} ]")

    log.info("End")
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
