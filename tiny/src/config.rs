use serde::{Deserialize, Serialize};
use std::{error::Error, fs, path::PathBuf, str::FromStr};

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub pull_from: String,
    pub push_to: String,

    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub density: f64,

    pub learning_rate: f64,
    pub steps: usize,
    pub retain: usize,
}

impl Config {
    pub fn load(path: &str) -> Result<Config, Box<dyn Error>> {
        let path = PathBuf::from_str(path)?;

        let config_bytes = fs::read(path)?;
        let config = serde_json::from_slice(&config_bytes)?;

        Ok(config)
    }

    pub fn new() -> Config {
        Config {
            pull_from: "tcp://127.0.0.1:1201".into(),
            push_to: "tcp://127.0.0.1:1202".into(),

            size: 24,
            d_in: 1,
            d_out: 1,
            density: 0.8,

            learning_rate: 0.001,
            steps: 50,
            retain: 1000,
        }
    }

    pub fn dump(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let path = PathBuf::from_str(path)?;
        let config_str = serde_json::to_string_pretty(&self)?;
        fs::write(path, config_str)?;

        Ok(())
    }
}
