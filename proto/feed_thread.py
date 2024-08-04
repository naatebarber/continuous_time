import json
import traceback
from threading import Thread
from typing import Any, Callable, Dict, TypedDict

import zmq


class Frame(TypedDict):
    ts: float
    channels: Dict[str, float]


class FeedThread(Thread):
    def __init__(self, address: str, topic: str, handler: Callable[[Frame], Any]):
        super().__init__()

        context = zmq.Context()
        self.sub_sock = context.socket(zmq.SUB)
        self.sub_sock.subscribe(topic.encode("utf-8"))
        self.sub_sock.connect(address)

        self.poller = zmq.Poller()
        self.poller.register(self.sub_sock, zmq.POLLIN)

        self.handler = handler
        self.lifetime = 0

    def run(self):
        while True:
            socks = dict(self.poller.poll(10))

            if self.sub_sock in socks:
                _, _, msgb = self.sub_sock.recv_multipart()
                frame: Frame = {**json.loads(msgb.decode("utf-8"))}
                self.handler(frame, self.lifetime)
                self.lifetime += 1
