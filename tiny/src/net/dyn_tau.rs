use ndarray::{Array1, Array2};
use rand::{prelude::*, thread_rng, Rng};
use std::collections::VecDeque;

pub enum NeuronType {
    Input,
    Hidden,
    Output,
}

#[derive(Clone)]
pub struct NeuronFrame {
    pub state: f64,
    pub instant: f64,
    pub loss: f64,
}

pub struct Neuron {
    pub state: f64,
    pub neuron_type: NeuronType,
}

impl Neuron {
    pub fn new() -> Neuron {
        Neuron {
            state: 0.,
            neuron_type: NeuronType::Hidden,
        }
    }

    pub fn sigmoid(x: f64) -> f64 {
        let clipped_x = f64::max(f64::min(500., x), -500.);
        1. / (1. + f64::exp(-clipped_x))
    }

    pub fn dsigmoid(x: f64) -> f64 {
        Neuron::sigmoid(x) * (1. - Neuron::sigmoid(x))
    }

    pub fn next_state(
        &self,
        connections: Vec<(*const Neuron, f64)>,
        external_influence: f64,
    ) -> f64 {
        unsafe {
            -self.state
                + (connections
                    .iter()
                    .filter(|(_, w)| *w != 0.0)
                    .map(|(neuron, weight)| Neuron::sigmoid((**neuron).state) * weight)
                    .sum::<f64>())
                + external_influence
        }
    }

    pub fn euler_step(
        &mut self,
        connections: Vec<(*const Neuron, f64)>,
        tau: f64,
        instant: f64,
        external_influence: f64,
    ) -> NeuronFrame {
        self.state += tau * self.next_state(connections, external_influence);

        return NeuronFrame {
            state: self.state,
            instant,
            loss: 0.,
        };
    }
}

pub struct DynTauNetwork {
    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub density: f64,
    pub desired_connections: f64,

    pub neurons: Vec<Neuron>,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,

    pub weights: Array2<f64>,
    pub taus: Array1<f64>,

    pub forward_invocations: Array1<usize>,
    pub cache: Vec<VecDeque<NeuronFrame>>,

    pub instant: f64,
    pub initialized: bool,
}

impl DynTauNetwork {
    pub fn new(size: usize, d_in: usize, d_out: usize) -> DynTauNetwork {
        let mut cache = Vec::with_capacity(size);
        cache.fill(VecDeque::new());

        DynTauNetwork {
            size,
            d_in,
            d_out,
            density: 0.,
            desired_connections: 0.,

            neurons: (0..size).map(|_| Neuron::new()).collect::<Vec<Neuron>>(),
            input_neurons: Vec::new(),
            output_neurons: Vec::new(),

            weights: Array2::zeros((size, size)),
            taus: Array1::zeros(size),

            forward_invocations: Array1::zeros(size),
            cache,

            instant: 0.,
            initialized: false,
        }
    }

    pub fn init_weight(&self, bound: Option<f64>, rng: &mut ThreadRng) -> f64 {
        if let Some(bound) = bound {
            rng.gen_range(-bound..bound)
        } else {
            let bound = f64::sqrt(6.) / (self.d_in + self.d_out) as f64;
            rng.gen_range(-bound..bound)
        }
    }

    pub fn weave(&mut self, density: f64) {
        self.density = density;
        let mut rng = thread_rng();

        let max_connections = self.size.pow(2);
        let mut desired_connections = (max_connections as f64 * self.density).floor() as usize;

        self.neurons = (0..self.size)
            .map(|_| Neuron::new())
            .collect::<Vec<Neuron>>();

        let ix_matrix = (0..self.size)
            .map(|y| {
                (0..self.size)
                    .map(move |x| (y, x))
                    .collect::<Vec<(usize, usize)>>()
            })
            .collect::<Vec<Vec<(usize, usize)>>>();

        let mut flat_ixlist = Vec::new();
        ix_matrix
            .into_iter()
            .for_each(|mut row| flat_ixlist.append(&mut row));
        flat_ixlist.shuffle(&mut rng);

        while desired_connections > 0 {
            let (y, x) = match flat_ixlist.pop() {
                Some(ix) => ix,
                None => continue,
            };

            self.weights[[y, x]] = self.init_weight(None, &mut rng);
            desired_connections -= 1;
        }

        let mut neuron_ixlist = (0..self.size).collect::<Vec<usize>>();
        neuron_ixlist.shuffle(&mut rng);

        self.input_neurons = neuron_ixlist.drain(0..self.d_in).collect::<Vec<usize>>();
        self.output_neurons = neuron_ixlist.drain(0..self.d_out).collect::<Vec<usize>>();

        for neuron_ix in self.output_neurons.iter() {
            let output_neuron = &mut self.neurons[*neuron_ix];
            output_neuron.neuron_type = NeuronType::Output;
        }

        for neuron_ix in self.input_neurons.iter() {
            let input_neuron = &mut self.neurons[*neuron_ix];
            input_neuron.neuron_type = NeuronType::Input;
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>, instant: f64) -> Option<Vec<f64>> {
        if self.initialized == false {
            self.instant = instant;
            self.initialized = true;
        }

        let mut forward_cache = Array1::zeros(self.size);
        let mut forward_invocations: Array1<usize> = Array1::zeros(self.size);

        let smallest_tau_step_size = self.taus.iter().fold(f64::MAX, |a, v| f64::min(a, *v));

        while self.instant < instant {
            let mut input_ix = 0;

            let neuron_pointers = self
                .neurons
                .iter()
                .map(|n| n as *const Neuron)
                .collect::<Vec<*const Neuron>>();

            for (neuron_ix, neuron) in self.neurons.iter_mut().enumerate() {
                forward_cache[neuron_ix] += smallest_tau_step_size;

                if forward_cache[neuron_ix] < self.taus[neuron_ix] {
                    continue;
                } else {
                    forward_cache[neuron_ix] = 0.;
                }

                let connections = self
                    .weights
                    .row(neuron_ix)
                    .iter()
                    .enumerate()
                    .map(|(other_neuron_ix, weight)| {
                        (neuron_pointers[other_neuron_ix] as *const Neuron, *weight)
                    })
                    .collect::<Vec<(*const Neuron, f64)>>();

                let frame = match neuron.neuron_type {
                    NeuronType::Input => {
                        let frame = neuron.euler_step(
                            connections,
                            self.taus[neuron_ix],
                            self.instant,
                            inputs[input_ix],
                        );
                        input_ix += 1;
                        frame
                    }
                    _ => neuron.euler_step(connections, self.taus[neuron_ix], self.instant, 0.),
                };

                forward_invocations[neuron_ix] += 1;
                self.cache[neuron_ix].push_back(frame);
            }

            self.instant += smallest_tau_step_size;
        }

        self.forward_invocations = forward_invocations;

        let output_state = self
            .neurons
            .iter()
            .filter(|n| match n.neuron_type {
                NeuronType::Output => true,
                _ => false,
            })
            .map(|n| n.state)
            .collect::<Vec<f64>>();

        Some(output_state)
    }

    pub fn temporal_cache_latest(&self, backwards_cache: &Array1<usize>) -> usize {
        let mut latest_neuron = 0;
        let mut latest_time = 0.;

        for (neuron_ix, nc) in self.cache.iter().enumerate() {
            let frame = &nc[backwards_cache[neuron_ix]];
            if latest_time < frame.instant {
                latest_time = frame.instant;
                latest_neuron = neuron_ix
            }
        }

        return latest_neuron;
    }

    pub fn backward(&mut self, learning_rate: f64, losses: Vec<f64>) {
        let mut weight_gradient = Array2::<f64>::zeros((self.size, self.size));

        let neuron_pointers = self
            .neurons
            .iter()
            .map(|n| n as *const Neuron)
            .collect::<Vec<*const Neuron>>();

        let mut backward_cache = Array1::<usize>::zeros(self.size);
        self.cache
            .iter()
            .enumerate()
            .for_each(|(i, nc)| backward_cache[i] = nc.len() - 1);

        let mut delta = Array1::<f64>::zeros(self.size);

        while backward_cache.sum() > 0 {
            let neuron_ix = self.temporal_cache_latest(&backward_cache);
            let frame = &mut self.cache[neuron_ix][backward_cache[neuron_ix]];

            if self.output_neurons.contains(&neuron_ix) {
                // backfill loss to the most recently forwarded. other output neurons will already have loss from
                // a previous call of the following logic
                if self.forward_invocations[neuron_ix] > 0 {
                    match self
                        .output_neurons
                        .iter()
                        .position(|output_neuron_ix| *output_neuron_ix == neuron_ix)
                    {
                        Some(output_ix) => frame.loss = losses[output_ix],
                        None => (),
                    }
                    self.forward_invocations[neuron_ix] -= 1
                }

                delta[neuron_ix] += frame.loss;
            }

            for (neuron_ix, ..) in self.neurons.iter().enumerate() {
                if delta[neuron_ix] == 0. {
                    continue;
                }

                let connections = self
                    .weights
                    .row(neuron_ix)
                    .iter()
                    .enumerate()
                    .map(|(other_neuron_ix, weight)| {
                        (
                            other_neuron_ix,
                            neuron_pointers[other_neuron_ix] as *const Neuron,
                            *weight,
                        )
                    })
                    .collect::<Vec<(usize, *const Neuron, f64)>>();

                for (other_neuron_ix, .., weight) in connections {
                    if weight == 0. {
                        continue;
                    }

                    let other_frame = &self.cache[other_neuron_ix][backward_cache[other_neuron_ix]];

                    let grad = delta[neuron_ix] * Neuron::dsigmoid(other_frame.state);
                    weight_gradient[[neuron_ix, other_neuron_ix]] += grad;
                    delta[other_neuron_ix] +=
                        delta[neuron_ix] * weight * Neuron::dsigmoid(other_frame.state)
                }
            }

            backward_cache[neuron_ix] -= 1;
        }

        weight_gradient *= learning_rate;
        self.weights -= &weight_gradient;
    }

    pub fn drain(&mut self, retain_time: f64) {
        for neuron_cache in self.cache.iter_mut() {
            while neuron_cache.len() > 0 && neuron_cache[0].instant + retain_time < self.instant {
                neuron_cache.pop_front();
            }
        }
    }
}
