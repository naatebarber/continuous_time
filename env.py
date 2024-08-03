import json
import time
from collections import deque
from traceback import print_exc
from typing import Any, Callable, Dict, TypedDict

import zmq

from ctrnn_hybrid_bptt_rtrl import CTRNN, Neuron
from graph_thread import GraphThread

# For use with snail waitress noisy feed generator


class Frame(TypedDict):
    ts: float
    channels: Dict[str, float]


def env(handler: Callable[[Frame, int], Any]):
    context = zmq.Context()
    sub_sock = context.socket(zmq.SUB)
    sub_sock.subscribe(b"waitress")
    sub_sock.connect("tcp://127.0.0.1:1201")

    poller = zmq.Poller()
    poller.register(sub_sock, zmq.POLLIN)

    lifetime = 0

    while True:
        try:
            socks = dict(poller.poll())

            if sub_sock in socks:
                topic, channel, msgb = sub_sock.recv_multipart()
                print(topic.decode("utf-8"), channel.decode("utf-8"), "<---")
                msgj = json.loads(msgb.decode("utf-8"))
                handler(msgj, lifetime)
                lifetime += 1

        except Exception as e:
            print_exc(e)


def cyclical_time(window: float, current: float):
    m = current % window
    t = m / window
    return t


def relative_time(start_time: float, end_time: float, current: float):
    c = current - start_time
    e = end_time - start_time
    t = c / e
    return t


def main():
    tiny = CTRNN(8, 1, 1).weave()
    small = CTRNN(12, 1, 1).weave()
    medium = CTRNN(20, 1, 1).weave()
    large = CTRNN(40, 1, 1).weave()

    window = 3600

    taus = deque()
    series = deque()

    losses_tiny = deque()
    losses_small = deque()
    losses_medium = deque()
    losses_large = deque()

    gt = GraphThread([losses_tiny, losses_small, losses_medium, losses_large])

    gt.daemon = True
    gt.start()

    global start_time
    global end_time
    start_time = time.time()
    end_time = start_time + 3600

    def handler(event: Frame, lifetime: int):
        ts = event.get("ts")
        data = event.get("channels").get("waitress")

        taus.append(relative_time(start_time, end_time, ts))
        series.append(data)

        if len(series) < 10:
            print("accumulating data")
            return

        train_current_data = series[0]
        train_target_tau = taus[1]
        train_target_data = series[1]

        model_args = dict(
            inputs=[train_current_data],
            target=[train_target_data],
            step_size=0.001,
            next_tau=train_target_tau,
            learning_rate=0.1,
        )

        losses_tiny.append((lifetime, tiny.step(**model_args) + 10))
        losses_small.append((lifetime, small.step(**model_args) + 20))
        losses_medium.append((lifetime, medium.step(**model_args) + 30))
        losses_large.append((lifetime, large.step(**model_args)))

        while len(series) > 10:
            series.popleft()

        while len(taus) > 10:
            taus.popleft()

    env(handler)


main()
