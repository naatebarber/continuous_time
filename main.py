from collections import deque
from typing import Dict

from ctrnn2 import CTRNN
from feed_thread import FeedThread, Frame
from graph_thread import GraphThread

if __name__ == "__main__":
    networks = {"small": CTRNN(size=12, d_in=1, d_out=1, density=0.7)}

    global lifetime
    global targets
    global losses
    global outputs

    targets = deque()
    losses = {k: deque() for k in networks.keys()}
    outputs = {k: deque() for k in networks.keys()}

    def feed_handler(frame: Frame, lifetime: int):
        tau = frame.get("ts")
        data = frame.get("channels").get("waitress")
        targets.append([data])

        if len(targets) < 2:
            return

        learning_rate = 0.0001
        steps = 30
        retain = 1000

        for network_name, network in networks.items():
            network_state = network.step(
                inputs=targets[-2],
                to_tau=tau,
                steps=steps,
                target=targets[-1],
                learning_rate=learning_rate,
                retain=retain,
            )

            if network_state is None:
                continue

            output, loss = network_state

            outputs[network_name].append((lifetime, output[0]))
            losses[network_name].append((lifetime, loss))

    graph_thread = GraphThread(iters=losses.values())
    feed_thread = FeedThread(
        address="tcp://127.0.0.1:1201", topic="waitress", handler=feed_handler
    )

    feed_thread.daemon = True
    feed_thread.start()

    # run blocking because matplotlib complains if it's not the main thread. smh.
    graph_thread.run_blocking()
    feed_thread.join()
