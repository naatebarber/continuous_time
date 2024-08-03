from threading import Thread
from typing import Iterable
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class GraphThread(Thread):
    def __init__(self, iter: deque):
        super().__init__()
        self.iter = iter

        self.mnx = 0
        self.mxx = 0
        self.mny = 0
        self.mxy = 0

    def run(self):
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        (ln,) = plt.plot([], [], "b-", animated=True)

        def init():
            ax.set_xlim(self.mnx, self.mxx)
            ax.set_ylim(self.mny, self.mxy)
            ln.set_data([], [])
            return (ln,)

        def update(frame):
            print("update")
            while len(self.iter) > 0:
                frame = self.iter.popleft()
                x = frame[0]
                y = frame[1]

                self.mnx = min(self.mnx, x)
                self.mxx = max(self.mxx, x)
                self.mny = min(self.mny, y)
                self.mxy = max(self.mxy, y)

                xdata.append(frame[0])
                ydata.append(frame[1])

            clxmn, clxmx = ax.get_xlim()
            clymn, clymx = ax.get_ylim()

            if (
                self.mnx < clxmn
                or self.mxx > clxmx
                or self.mny < clymn
                or self.mxy > clymx
            ):
                ax.set_xlim(2 * self.mnx, 2 * self.mxx)
                ax.set_ylim(2 * self.mny, 2 * self.mxy)
                fig.canvas.draw()

            ln.set_data(xdata, ydata)

            return (ln,)

        a = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=30)
        plt.show()
