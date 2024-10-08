import math
import random
from collections import deque
from typing import Dict, List

import numpy as np


class Neuron:
    def __init__(self):
        self.state = None
        self.tau = None
        self.nexto: Dict[Neuron, float] = {}

        self.state_history = deque()

    def bind(self, neu, weight: float):
        new_c = neu not in self.nexto
        self.nexto[neu] = weight
        return new_c

    def sync(self, initial_state: float, initial_tau: float):
        self.state = initial_state
        self.tau = initial_tau

    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x: float):
        sigmoid_x = Neuron.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def next_state(self, extern: float):
        return (
            -self.state
            + sum(
                [
                    weight * Neuron.sigmoid(neu.state)
                    for neu, weight in self.nexto.items()
                ]
            )
            + extern
        )

    def euler_step(self, next_tau: float, external_t: float = 0):
        step_size = next_tau - self.tau
        ds_dt = self.next_state(extern=external_t)
        self.state += step_size * ds_dt
        self.tau = next_tau

        self.state_history.appendleft(self.state)

    def dump(self):
        dump_state_hist = list([*self.state_history])
        self.state_history = deque()
        return Neuron.sigmoid(self.state), dump_state_hist


class CTRNN:
    def __init__(self, size: int, input_features: int, output_features: int):
        if input_features + output_features > size:
            raise Exception(
                "size must be larger than input and output features combined."
            )

        self.size = size
        self.neurons: List[Neuron] = []

        self.init_bound = 0

        self.input_features = input_features
        self.output_features = output_features
        self.inf: List[Neuron] = []
        self.ouf: List[Neuron] = []

        self.tau = None

    def set_tau(self, tau: float):
        self.tau = tau

    def xavier(self):
        self.init_bound = self.init_bound or math.sqrt(6) / math.sqrt(
            self.input_features + self.output_features
        )
        return (np.random.random_sample() * 2 * self.init_bound) - self.init_bound

    def weave(self, density: float = 0.5):
        max_connections = self.size * (self.size - 1)

        connections = density * max_connections

        if density > 1:
            raise Exception(
                "desired density %s greater than the maximum possible density 1"
                % (density)
            )

        self.neurons = [Neuron() for _ in range(self.size)]

        def shuffled_neurons():
            neurons = [*self.neurons]
            random.shuffle(neurons)
            return neurons

        assignment_map: Dict[Neuron, List[Neuron]] = {
            neuron: shuffled_neurons() for neuron in self.neurons
        }

        c = 0
        while c < connections:
            for neuron in self.neurons:
                neuron.bind(assignment_map[neuron].pop(), self.xavier())
                c += 1

                if c > connections:
                    break

        del assignment_map

        in_out_set = shuffled_neurons()
        while len(self.inf) < self.input_features:
            self.inf.append(in_out_set.pop())
        while len(self.ouf) < self.output_features:
            self.ouf.append(in_out_set.pop())

        return self

    def forward(self, inputs: List[float], step_size: float, current_time: float):
        if len(self.inf) != len(inputs):
            raise Exception(
                "size mismatch, input neurons: %s, input data shape: %s"
                % (len(self.inf), len(inputs))
            )

        if not self.tau:
            print("baselining network...")

            print("syncing tau...")
            self.tau = current_time

            print("syncing neurons...")
            inp_ix = 0
            for neuron in self.neurons:
                if neuron in self.inf:
                    neuron.sync(inputs[inp_ix], self.tau)
                    inp_ix += 1
                else:
                    neuron.sync(np.random.uniform(-1, 1), self.tau)
            return

        while self.tau < current_time:
            inp_ix = 0
            for ix, neuron in enumerate(self.neurons):
                if neuron in self.inf:
                    neuron.euler_step(self.tau, external_t=inputs[inp_ix])
                    inp_ix += 1
                else:
                    neuron.euler_step(self.tau, external_t=0)

            self.tau += step_size

        # (where i am, how i got here)
        return [neu.state for neu in self.ouf]

    def backward(self, target: List[float], learning_rate: float):
        if len(self.ouf) != len(target):
            raise Exception(
                "size mismatch, output neurons: %s, target data shape: %s"
                % (len(self.ouf), len(target))
            )

        num_steps = len(self.ouf[0].state_history)

        loss_fn = lambda pred, targ: (pred - targ) ** 2
        loss = [
            loss_fn(Neuron.sigmoid(neu.state), target[i])
            for i, neu in enumerate(self.ouf)
        ]

        for t in range(num_steps):
            for i, neu in enumerate(self.ouf):
                output_grad = loss[i] * Neuron.dsigmoid(neu.state_history[t])
                for other_neu in neu.nexto:
                    other_neu_gradient = output_grad * other_neu.state_history[t]
                    # Gradient descent
                    neu.nexto[other_neu] -= learning_rate * other_neu_gradient

        return np.mean(loss)

    def step(
        self,
        inputs: List[float],
        target: List[float],
        step_size: float,
        next_tau: float,
        learning_rate: float,
    ):
        self.forward(inputs=inputs, step_size=step_size, current_time=next_tau)
        loss = self.backward(target=target, learning_rate=learning_rate)

        [neu.dump() for neu in self.neurons]

        return loss
