use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct StepFrame {
    pub inputs: Vec<f64>,
    pub tau: f64,
    pub targets: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InfFrame {
    pub inputs: Vec<f64>,
    pub tau: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OutputFrame {
    pub outputs: Vec<f64>,
    pub tau: f64,
    pub loss: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DrainFrame {}
