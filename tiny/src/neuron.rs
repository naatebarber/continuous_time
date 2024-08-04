use std::collections::{HashMap, VecDeque};

pub enum NeuronType {
    Input,
    Hidden,
    Output,
}

pub struct Neuron {
    pub state: f64,
    pub tau: f64,
    pub states: VecDeque<f64>,
    pub taus: VecDeque<f64>,
    pub targets: Option<VecDeque<f64>>,

    pub neuron_type: NeuronType,
    pub connections: HashMap<*const Neuron, f64>,
}

impl Neuron {
    pub fn new() -> Neuron {
        Neuron {
            state: 0.,
            tau: 0.,
            states: VecDeque::new(),
            taus: VecDeque::new(),
            targets: None,

            neuron_type: NeuronType::Hidden,
            connections: HashMap::new(),
        }
    }

    pub fn set_output(&mut self) -> &mut Self {
        self.neuron_type = NeuronType::Output;
        self.targets = Some(VecDeque::new());
        self
    }

    pub fn set_input(&mut self) -> &mut Self {
        self.neuron_type = NeuronType::Input;
        self.targets = None;
        self
    }

    pub fn set_hidden(&mut self) -> &mut Self {
        self.neuron_type = NeuronType::Hidden;
        self.targets = None;
        self
    }

    pub fn sigmoid(x: f64) -> f64 {
        let clipped_x = f64::max(f64::min(500., x), -500.);
        1. / (1. + f64::exp(-clipped_x))
    }

    pub fn dsigmoid(x: f64) -> f64 {
        Neuron::sigmoid(x) * (1. - Neuron::sigmoid(x))
    }

    pub fn next_state(&self, external_influence: f64) -> f64 {
        unsafe {
            -self.state
                + (self
                    .connections
                    .iter()
                    .map(|(neuron, weight)| (**neuron).state * weight)
                    .sum::<f64>())
                + external_influence
        }
    }

    pub fn euler_step(&mut self, next_tau: f64, external_influence: f64) {
        let step_size = next_tau - self.tau;
        self.state += step_size * self.next_state(external_influence);
        self.tau = next_tau;

        self.states.push_back(self.state);
        self.taus.push_back(self.tau);
    }

    pub fn cache_target(&mut self, target: f64) -> &mut Self {
        if let Some(targets) = &mut self.targets {
            targets.push_back(target);
        }
        self
    }

    pub fn sync<T>(a: &mut VecDeque<T>, b: &mut VecDeque<T>) {
        let (longer, shorter) = match a.len() > b.len() {
            true => (a, b),
            false => (b, a),
        };

        while longer.len() > shorter.len() {
            longer.pop_front();
        }
    }

    pub fn drain(&mut self, retain: usize) -> usize {
        if self.taus.len() != self.states.len() {
            println!("(tiny) states:taus mismatch, attempting sync...");
            Neuron::sync(&mut self.taus, &mut self.states)
        }

        if let Some(targets) = &mut self.targets {
            if targets.len() != self.states.len() {
                println!("(tiny) targets:(states:taus) mismatch, attempting sync...");
                Neuron::sync(targets, &mut self.states);
                Neuron::sync(targets, &mut self.taus);
            }
        }

        while self.states.len() > retain {
            self.states.pop_front();
            self.taus.pop_front();

            if let Some(targets) = &mut self.targets {
                targets.pop_front();
            }
        }

        return self.states.len();
    }
}
