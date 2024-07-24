import asyncio
from asyncio import WindowsSelectorEventLoopPolicy

import zmq
import zmq.asyncio

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
ctx = zmq.asyncio.Context()
url = "tcp://localhost:5555"


async def async_process(msg):
    print(msg)


async def recv_and_process():
    sock = ctx.socket(zmq.PULL)
    sock.bind(url)
    msg = await sock.recv_multipart()  # waits for msg to be ready
    reply = await async_process(msg)
    await sock.send_multipart(reply)


print("asyncio.run")
asyncio.run(recv_and_process())

print("done")


