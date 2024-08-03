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
        self.tau_history = deque()

    def bind(self, neu, weight: float):
        new_c = neu not in self.nexto
        self.nexto[neu] = weight
        return new_c

    def sync(self, initial_state: float, initial_tau: float):
        self.state = initial_state
        self.tau = initial_tau

    @staticmethod
    def sigmoid(x: float):
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))

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
        self.tau_history.appendleft(self.tau)


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

        self.target_history: List[List[float]] = []
        self.tau_history: List[float] = []

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

    def forward(self, inputs: List[float], steps: int, current_time: float):
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

        step_size = (current_time - self.tau) / steps

        while self.tau < current_time:
            inp_ix = 0
            for ix, neuron in enumerate(self.neurons):
                if neuron in self.inf:
                    neuron.euler_step(self.tau, external_t=inputs[inp_ix])
                    inp_ix += 1
                else:
                    neuron.euler_step(self.tau, external_t=0)

            self.tau += step_size

        return [neu.state for neu in self.ouf]

    def get_targets(self, spot_tau: float):
        target_ix = len(self.target_history) - 1
        for i in reversed(range(len(self.tau_history))):
            if self.tau_history[i] > spot_tau:
                target_ix = i
            else:
                break

        return self.target_history[target_ix]

    def backward(self, target: List[float], learning_rate: float):
        if len(self.ouf) != len(target):
            raise Exception(
                "size mismatch, output neurons: %s, target data shape: %s"
                % (len(self.ouf), len(target))
            )

        weight_gradients = {neu: {} for neu in self.neurons}

        for neu in self.neurons:
            for other_neu in neu.nexto:
                weight_gradients[neu][other_neu] = 0.0

        # todo ensure all neurons state history and tau history are the same length
        num_steps = len(self.ouf[0].state_history)

        for t in range(num_steps):
            delta = {neu: 0.0 for neu in self.neurons}

            targets = self.get_targets(self.ouf[0].tau_history[t])

            for i, neu in enumerate(self.ouf):
                # TODO target is now always reflecting last state.
                output_grad = 2 * (neu.state_history[t] - targets[i])
                delta[neu] = output_grad

            for neu in self.neurons:
                if delta[neu] == 0.0:
                    continue

                for other_neu, weight in neu.nexto.items():
                    weight_grad = delta[neu] * Neuron.sigmoid(
                        other_neu.state_history[t]
                    )
                    weight_gradients[neu][other_neu] += weight_grad
                    delta[other_neu] += (
                        delta[neu]
                        * weight
                        * Neuron.dsigmoid(other_neu.state_history[t])
                    )

        for neu in self.neurons:
            for other_neu, gradient in weight_gradients[neu].items():
                neu.nexto[other_neu] -= learning_rate * gradient

    def step(
        self,
        inputs: List[float],
        next_tau: float,
        steps: int,
        target: List[float],
        learning_rate: float,
    ):
        self.target_history.append(target)
        self.tau_history.append(next_tau)

        self.forward(inputs=inputs, steps=steps, current_time=next_tau)
        loss = self.backward(target=target, learning_rate=learning_rate)

        for neu in self.neurons:
            while len(neu.state_history) > steps * 10:
                neu.state_history.pop()

        return loss
