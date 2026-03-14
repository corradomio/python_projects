import sys
import threading
import time

print(f"Python version: {sys.version}")
print(f"Version info: {sys.version_info}")


def crawl(link, delay=3):
    print(f"crawl started for {link}")
    time.sleep(delay)  # Blocking I/O (simulating a network request)
    print(f"crawl ended for {link}")

links = [
    "https://python.org",
    "https://docs.python.org",
    "https://peps.python.org",
]

print("prepare")
# Start threads for each link
threads = []
for link in links:
    # Using `args` to pass positional arguments and `kwargs` for keyword arguments
    t = threading.Thread(target=crawl, args=(link,), kwargs={"delay": 2})
    threads.append(t)

print("start")
# Start each thread
for t in threads:
    t.start()

print("wait for completion")
# Wait for all threads to finish
for t in threads:
    t.join()

print("done")