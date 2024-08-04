use std::collections::{HashMap, VecDeque};

use crate::neuron::{Neuron, NeuronType};
use rand::{prelude::*, thread_rng, Rng};

pub struct Network {
    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub density: f64,
    pub desired_connections: f64,

    pub neurons: Vec<Neuron>,
    pub input_neurons: Vec<*mut Neuron>,
    pub output_neurons: Vec<*mut Neuron>,

    pub tau: f64,
    pub steps: usize,

    pub initialized: bool,
    pub lifecycle: usize,
}

impl Network {
    pub fn new<'a>(size: usize, d_in: usize, d_out: usize) -> Network {
        Network {
            size,
            d_in,
            d_out,
            density: 0.,

            neurons: Vec::new(),
            input_neurons: Vec::new(),
            output_neurons: Vec::new(),

            desired_connections: 0.,
            tau: 0.,
            steps: 0,

            initialized: false,
            lifecycle: 0,
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

        for _ in 0..self.size {
            self.neurons.push(Neuron::new());
        }

        let mut input_output_neuron_pointers: Vec<*mut Neuron> = self
            .neurons
            .iter_mut()
            .map(|n| n as *mut Neuron)
            .collect::<Vec<*mut Neuron>>();

        input_output_neuron_pointers.shuffle(&mut rng);

        while self.input_neurons.len() < self.d_in && input_output_neuron_pointers.len() > 0 {
            match input_output_neuron_pointers.pop() {
                Some(np) => self.input_neurons.push(np),
                _ => (),
            }
        }

        while self.output_neurons.len() < self.d_out && input_output_neuron_pointers.len() > 0 {
            match input_output_neuron_pointers.pop() {
                Some(np) => self.output_neurons.push(np),
                _ => (),
            }
        }

        let mut shuffled_neuron_matrix = self
            .neurons
            .iter()
            .map(|_| {
                let mut neuron_references = self
                    .neurons
                    .iter()
                    .map(|neuron_ref| {
                        let neuron_pt: *const Neuron = neuron_ref as *const Neuron;
                        neuron_pt
                    })
                    .collect::<Vec<*const Neuron>>();
                neuron_references.shuffle(&mut rng);
                return neuron_references;
            })
            .collect::<Vec<Vec<*const Neuron>>>();

        let mut shuffled_neuron_ixlist = (0..self.neurons.len()).collect::<Vec<usize>>();
        shuffled_neuron_ixlist.shuffle(&mut rng);

        while desired_connections > 0 {
            match shuffled_neuron_ixlist.choose(&mut rng) {
                Some(neuron_ix) => match shuffled_neuron_matrix[*neuron_ix].pop() {
                    Some(other_neuron) => {
                        let other_neuron_pointer: *const Neuron = other_neuron as *const Neuron;
                        let other_neuron_weight = self.init_weight(None, &mut rng);

                        self.neurons[*neuron_ix]
                            .connections
                            .insert(other_neuron_pointer, other_neuron_weight);
                        desired_connections -= 1;
                    }
                    None => continue,
                },
                None => continue,
            }
        }
    }

    pub fn init(&mut self, inputs: Vec<f64>, next_tau: f64) {
        if self.initialized == false {
            self.tau = next_tau;

            for neuron in self.neurons.iter_mut() {
                neuron.tau = next_tau;
            }

            unsafe {
                for output_neuron in self.output_neurons.iter() {
                    (**output_neuron).targets = Some(VecDeque::new());
                    (**output_neuron).neuron_type = NeuronType::Output;
                }

                for (i, input_neuron) in self.input_neurons.iter().enumerate() {
                    (**input_neuron).state = inputs[i];
                    (**input_neuron).neuron_type = NeuronType::Input;
                }
            }

            self.initialized = true;
        }
    }

    pub fn forward(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Option<Vec<f64>>,
    ) -> Option<Vec<f64>> {
        if !self.initialized {
            self.init(inputs, next_tau);
            return None;
        }

        let step_size = (next_tau - self.tau) / steps as f64;

        while self.tau < next_tau {
            let mut input_ix = 0;
            let mut target_ix = 0;

            for neuron in self.neurons.iter_mut() {
                match neuron.neuron_type {
                    NeuronType::Hidden => {
                        neuron.euler_step(self.tau, 0.);
                    }
                    NeuronType::Input => {
                        neuron.euler_step(self.tau, inputs[input_ix]);
                        input_ix += 1;
                    }
                    NeuronType::Output => {
                        neuron.euler_step(self.tau, 0.);
                        if let Some(targets) = &targets {
                            neuron.cache_target(targets[target_ix]);
                            target_ix += 1;
                        }
                    }
                }
            }

            self.tau += step_size;
        }

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

    pub fn backward(&mut self, steps: usize, learning_rate: f64) -> Option<f64> {
        let mut weight_gradient: HashMap<*const Neuron, HashMap<*const Neuron, f64>> =
            HashMap::new();

        let mut losses: Vec<Vec<f64>> = (0..self.neurons.len())
            .map(|_| vec![])
            .collect::<Vec<Vec<f64>>>();

        unsafe {
            for t in (0..steps).rev() {
                let mut delta: HashMap<*const Neuron, f64> = HashMap::new();

                for (i, output_neuron) in self.output_neurons.iter().enumerate() {
                    let state_t = (**output_neuron).states[t];
                    let target_t = match &(**output_neuron).targets {
                        Some(targets) => targets[t],
                        None => {
                            println!("(tiny) no target state on output neuron during BPTT...");
                            return None;
                        }
                    };

                    let error = state_t - target_t;
                    losses[i].push(error.powi(2));

                    let grad = 2. * error;
                    delta.insert(*output_neuron as *const Neuron, grad);
                }

                for neuron in self.neurons.iter() {
                    let np = neuron as *const Neuron;

                    let neuron_delta = match delta.get(&np) {
                        Some(x) => x.clone(),
                        None => continue,
                    };

                    for (other_neuron, weight) in neuron.connections.iter() {
                        let grad = neuron_delta * Neuron::dsigmoid((**other_neuron).states[t]);

                        if !weight_gradient.contains_key(&np) {
                            weight_gradient.insert(np, HashMap::new());
                        }

                        match weight_gradient.get_mut(&np) {
                            Some(ng) => match ng.get_mut(other_neuron) {
                                Some(existing_gradient) => {
                                    *existing_gradient += grad;
                                }
                                None => {
                                    ng.insert(*other_neuron, grad);
                                }
                            },
                            _ => (),
                        };

                        if !delta.contains_key(other_neuron) {
                            delta.insert(*other_neuron, 0.);
                        }

                        match delta.get_mut(other_neuron) {
                            Some(d) => {
                                *d += neuron_delta
                                    * weight
                                    * Neuron::dsigmoid((**other_neuron).states[t])
                            }
                            None => (),
                        }
                    }
                }
            }

            self.neurons.iter_mut().for_each(|neuron| {
                let np = neuron as *const Neuron;
                match weight_gradient.get(&np) {
                    Some(g) => {
                        g.iter().for_each(|(other_neuron, gradient)| {
                            let lr_gradient = learning_rate * gradient;

                            match neuron.connections.get_mut(other_neuron) {
                                Some(weight) => *weight -= lr_gradient,
                                None => (),
                            }
                        });
                    }
                    _ => (),
                }
            });
        }

        let mean_neuron_losses = losses
            .into_iter()
            .filter(|neuron_loss| neuron_loss.len() > 0)
            .map(|neuron_loss| neuron_loss.iter().sum::<f64>() / neuron_loss.len() as f64)
            .collect::<Vec<f64>>();
        let mean_loss = mean_neuron_losses.iter().sum::<f64>() / mean_neuron_losses.len() as f64;

        Some(mean_loss)
    }

    pub fn step(
        &mut self,
        inputs: Vec<f64>,
        next_tau: f64,
        steps: usize,
        targets: Vec<f64>,
        retain: usize,
        learning_rate: f64,
    ) -> Option<(Vec<f64>, f64)> {
        self.lifecycle += 1;

        let network_output = match self.forward(inputs, next_tau, steps, Some(targets)) {
            Some(os) => os,
            None => return None,
        };

        let cross_neuron_bptt_steps = self
            .neurons
            .iter_mut()
            .map(|neuron| neuron.drain(retain))
            .filter(|state_length| *state_length > 0)
            .collect::<Vec<usize>>();

        let reference_steps = &cross_neuron_bptt_steps[0];
        let state_synchronized = cross_neuron_bptt_steps
            .iter()
            .all(|e| *e == *reference_steps);

        if !state_synchronized {
            panic!("(tiny) neurons fell out of sync!");
        }

        let loss = match self.backward(*reference_steps, learning_rate) {
            Some(l) => l,
            None => return None,
        };

        return Some((network_output, loss));
    }
}
