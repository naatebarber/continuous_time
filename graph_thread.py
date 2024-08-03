from collections import deque
from threading import Thread
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class GraphThread(Thread):
    def __init__(self, iters: List[deque]):
        super().__init__()
        self.iters = iters

        self.mnx = 0
        self.mxx = 0
        self.mny = 0
        self.mxy = 0

        self.colors = ["red", "green", "blue", "purple", "orange", "pink", "black"]

    def pop_color(self):
        return self.colors.pop(0)

    def run(self):
        fig, ax = plt.subplots()
        xdata, ydata = [], []

        datas = [[[], []] for _ in self.iters]

        lns = [
            plt.plot([], [], "b-", animated=True, color=self.pop_color())[0]
            for _ in self.iters
        ]

        def init():
            ax.set_xlim(self.mnx, self.mxx)
            ax.set_ylim(self.mny, self.mxy)

            [ln.set_data([], []) for ln in lns]

            return [*lns]

        def update(frame):
            print("update")
            for ix, it in enumerate(self.iters):
                while len(it) > 0:
                    frame = it.popleft()
                    x = frame[0]
                    y = frame[1]

                    self.mnx = min(self.mnx, x)
                    self.mxx = max(self.mxx, x)
                    self.mny = min(self.mny, y)
                    self.mxy = max(self.mxy, y)

                    datas[ix][0].append(frame[0])
                    datas[ix][1].append(frame[1])

                    lns[ix].set_data(*datas[ix])

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

            return [*lns]

        a = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=30)
        plt.show()
