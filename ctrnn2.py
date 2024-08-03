from typing import List, Any, Dict, Callable
from collections import deque
import numpy as np
from numpy.typing import ArrayLike
import random


class Neuron:
    def __init__(self):
        self.state = None
        self.tau = None
        self.states = deque()
        self.taus = deque()
        self.targets = deque()

    def initialize(self, state: float, tau: float):
        self.state = state
        self.tau = tau

    @staticmethod
    def sigmoid(x: float):
        clipped_x = np.clip(-x, -500, 500)
        return 1 / (1 + np.exp(clipped_x))

    @staticmethod
    def dsigmoid(x: float):
        sigmoid_x = Neuron.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def next_state(
        self,
        weights: List[float],
        neurons: List[Any],
        external_influence: float,
        activation: Callable[[float], float],
    ):
        return (
            -self.state
            + sum(
                [
                    w * activation(neurons[i].state)
                    for i, w in enumerate(weights)
                    if w > 0
                ]
            )
            + external_influence
        )

    def euler_step(
        self,
        to_tau: float,
        weights: List[float],
        neurons: List[Any],
        external_influence: float,
    ):
        step_size = to_tau - self.tau
        ds_dt = self.next_state(weights, neurons, external_influence, Neuron.sigmoid)
        self.state += step_size * ds_dt
        self.tau = to_tau

        self.states.append(self.state)
        self.taus.append(self.tau)

    def cache_target(self, target_state: float):
        self.targets.append(target_state)

    def drain(self, retain: int):
        ejected_states = 0
        ejected_taus = 0

        while len(self.states) > retain:
            self.states.pop()
            ejected_states += 1
        while len(self.taus) > retain:
            self.taus.pop()
            ejected_taus += 1

        if ejected_taus != ejected_states:
            raise Exception("neuron memory is out of sync - state tau mismatch")

        return ejected_states


class CTRNN:
    def __init__(self, size: int, d_in: int, d_out: int, density: float):
        self.size = size
        self.d_in = d_in
        self.d_out = d_out
        self.density = density
        self.max_connections = size**2
        self.target_connections = self.max_connections * density

        self.neurons: List[Neuron] = []
        self.weights: ArrayLike = []
        self.input_neurons: List[Neuron] = []
        self.output_neurons: List[Neuron] = []

        self.tau = None
        self.targets = deque()
        self.target_taus = deque()
        self.target_steps: Dict[int, int] = {}

        self.total_steps = 0

        self.init_weights_matrix()

    def init_weight(self, custom_bound: float = None):
        bound = custom_bound or (np.sqrt(6) / np.sqrt(self.d_in + self.d_out))
        return np.random.uniform(-bound, bound)

    def init_weights_matrix(self):
        self.weights = np.zeros((self.size, self.size))
        assignable = list(
            np.array(
                [[(y, x) for x in range(self.size)] for y in range(self.size)]
            ).flatten()
        )
        random.shuffle(assignable)

        for _ in range(self.target_connections):
            y, x = assignable.pop()
            self.weights[y][x] = self.init_weight()

        self.neurons = [Neuron() for _ in range(self.size)]

        ix_pool = [x for x in range(len(self.neurons))]
        random.shuffle(ix_pool)

        in_ix = [ix_pool.pop() for _ in range(self.d_out)]
        out_ix = [ix_pool.pop() for _ in range(self.d_in)]
        self.input_neurons = [self.neurons[ix] for ix in sorted(in_ix)]
        self.output_neurons = [self.neurons[ix] for ix in sorted(out_ix)]

    def forward(
        self, inputs: List[float], to_tau: float, steps: float, target: float = 0
    ):
        if self.tau is None:
            self.tau = to_tau

            input_ix = 0
            for neuron in self.neurons:
                if neuron in self.input_neurons:
                    neuron.initialize(inputs[input_ix], self.tau)
                    input_ix += 1
                    continue
                neuron.initialize(0, self.tau)

            return

        step_size = (to_tau - self.tau) / steps
        while self.tau < to_tau:
            self.tau += step_size

            input_ix = 0
            output_ix = 0
            for i, neuron in enumerate(self.neurons):
                if neuron in self.input_neurons:
                    neuron.euler_step(
                        self.tau,
                        self.weights[i],
                        neurons=self.neurons,
                        external_influence=inputs[input_ix],
                    )
                    input_ix += 1
                    continue

                if neuron in self.output_neurons:
                    neuron.cache_target(target[output_ix])
                    output_ix += 1

                neuron.euler_step(self.tau, self.weights[i], self.neurons)

        return [neuron.state for neuron in self.output_neurons]

    def backward(self, learning_rate: float):
        weight_gradient = np.zeros((self.size, self.size))
        losses = [[] for _ in range(self.d_out)]

        for t in reversed(range(self.backward_steps)):
            delta = np.zeros((self.size))

            for i, output_neuron in enumerate(self.output_neurons):
                error = output_neuron.states[t] - output_neuron.targets[t]

                losses[i].append(error**2)
                output_grad = 2 * (error)

                neuron_ix = self.neurons.index(output_neuron)
                delta[neuron_ix] = output_grad

            for i in range(len(self.neurons)):
                if delta[i] == 0.0:
                    continue

                neuron_weights = self.weights[i]
                for j, weight in enumerate(neuron_weights):
                    grad = delta[i] * Neuron.dsigmoid(self.neurons[j].states[t])
                    weight_gradient[i][j] += grad
                    delta[j] += (
                        delta[i] * weight * Neuron.dsigmoid(self.neurons[j].states[t])
                    )

        weight_gradient *= learning_rate
        self.weights -= weight_gradient

        losses = [np.mean(l) for l in losses]
        loss = np.mean(losses)
        return loss

    def step(
        self,
        inputs: List[float],
        to_tau: float,
        steps: int,
        target: List[float],
        learning_rate: float,
        retain: int,
    ):
        self.targets.append(target)
        self.target_taus.append(to_tau)
        self.total_steps += steps

        network_output = self.forward(inputs, to_tau, steps, target)

        ejected = set([neuron.drain(retain) for neuron in self.neurons])
        if len(ejected) > 1:
            raise Exception(
                "network neuron memory mismatch - difference between neurons in number of ejected states"
            )
        ejected = ejected.pop()
        self.total_steps -= ejected

        network_loss = self.backward(learning_rate)

        return network_output, network_loss
