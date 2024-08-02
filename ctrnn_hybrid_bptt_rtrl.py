from typing import Dict, List
import math
import time
import numpy as np
import random
from collections import deque

class Neuron:
    def __init__(self, init_x: float, init_tau: float):
        self.state = init_x
        self.tau = init_tau
        self.nexto: Dict[Neuron, float] = {}

        self.state_history = deque()

    def bind(self, neu, weight: float):
        new_c = neu not in self.nexto
        self.nexto[neu] = weight
        return new_c

    def sigmoid(self, x: float):
        return 1 / (1 + math.exp(-x) )
    
    def dsigmoid(self, x: float):
        return self.sigmoid(x) * (math.exp(-x) * self.sigmoid(x))

    def next_state(self, extern: float):
        return -self.state + sum([ weight * self.sigmoid(neu.state) for neu, weight in self.nexto ]) + extern

    def integrate(self, next_tau: float, external_t: float = 0):
        step_size = next_tau - self.tau
        ds_dt = self.next_state(extern=external_t)
        self.state += step_size * ds_dt
        self.tau = next_tau

        self.state_history.appendleft(self.state)

    def dump(self):
        dump_state_hist = list([ *self.state_history ])
        self.state_history = deque()
        return self.sigmoid(self.state), dump_state_hist


class CTRNN:
    def __init__(self, size: int, input_features: int, output_features: int):
        if input_features + output_features > size:
            raise Exception("size must be larger than input and output features combined.")

        self.size = size
        self.neurons: List[Neuron] = []

        self.init_bound = 0

        self.input_features = input_features
        self.output_features = output_features
        self.inf: List[Neuron] = []
        self.ouf: List[Neuron] = []
        
        self.tau = time.time()

    def xavier(self):
        self.init_bound = self.init_bound or math.sqrt(6) / math.sqrt(self.input_features + self.output_features)
        return (np.random.random_sample() * 2 * self.init_bound) - self.init_bound

    def weave(self, connections: int):
        max_connections = self.size * (self.size - 1)
        if connections > max_connections:
            raise Exception("desired number of connections %s greater than the possible maximum %s for a network size of %s" % (connections, max_connections, self.size))

        c = 0

        lay = [ *self.neurons ]
        random.shuffle(lay)

        while c < connections:
            if len(lay) == 0:
                lay = [ *self.neurons ]
                random.shuffle(lay)
            
            neu = lay.pop()
            if neu.bind(random.choice(self.neurons), self.xavier()):
                c += 1

        lay = [ *self.neurons ]
        random.shuffle(lay)

        while len(self.inf) < self.input_features:
            self.inf.append(lay.pop())
        while len(self.ouf) < self.output_features:
            self.ouf.append(lay.pop())

    def forward(self, input: List[float], step_size: float, current_time: float):
        while self.tau < current_time:
            for ix, neuron in enumerate(self.neurons):
                if neuron in self.inf:
                    neuron.integrate(self.tau, external=input[ix])
                else:
                    neuron.integrate(self.tau, external=0)

            self.tau += step_size

        # (where i am, how i got here)
        return [ neu.sigmoid(neu.state) for neu in self.ouf ]
    
    def backward(self, target: List[float], learning_rate: float):
        num_steps = len(self.ouf[0].state_history)

        loss_fn = lambda pred, targ: pred - targ
        loss = [ 
            loss_fn(neu.sigmoid(neu.state), target[i]) 
            for i, neu in enumerate(self.ouf) 
        ]

        for t in range(num_steps):
            for i, neu in enumerate(self.ouf):
                output_grad = loss[i] * neu.dsigmoid(neu.state_history[t])
                for other_neu, weight in neu.nexto.items():
                    other_neu_gradient = output_grad * other_neu.state_history[t]
                    # Gradient descent
                    neu.nexto[other_neu] -= learning_rate * other_neu_gradient

    def step(self, inputs: List[float], target: List[float], step_size: float, next_tau: float, learning_rate: float):
        self.forward(input=inputs, step_size=step_size)
        self.backward(target=target, learning_rate=learning_rate)

        [
            neu.dump()
            for neu in self.neurons
        ]