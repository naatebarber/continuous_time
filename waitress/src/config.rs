use std::fs;
use std::str::FromStr;
use std::{error::Error, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Waves {
    Sinusoidal,
    Spike,
    SpikeDecay,
    Pwm,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WaveConfig {
    pub wave: Waves,
    pub offset: f64,
    pub period: usize,
}

#[derive(Serialize, Deserialize)]
pub struct WaitressConfig {
    pub pub_to: String,
    pub channel: String,
    pub sleep: f64,
    pub step: f64,
    pub offsetv: f64,
    pub minv: f64,
    pub maxv: f64,
    pub noise: f64,
    pub waves: Vec<WaveConfig>,
}

impl WaitressConfig {
    pub fn load(path: &str) -> Result<WaitressConfig, Box<dyn Error>> {
        let pb = PathBuf::from_str(path)?;
        let cfgb = fs::read(&pb)?;
        let cfg: WaitressConfig = serde_json::from_slice(&cfgb)?;

        Ok(cfg)
    }

    pub fn dump(path: &str) -> Result<(), Box<dyn Error>> {
        let cfg = WaitressConfig {
            pub_to: "tcp://127.0.0.1:1201".into(),
            channel: "waitress".into(),
            sleep: 1.,
            step: 1.,
            offsetv: 100.,
            minv: 0.,
            maxv: 10.,
            noise: 0.,
            waves: vec![
                WaveConfig {
                    wave: Waves::Sinusoidal,
                    offset: 0.,
                    period: 0,
                },
                WaveConfig {
                    wave: Waves::Spike,
                    offset: 5.,
                    period: 10,
                },
                WaveConfig {
                    wave: Waves::SpikeDecay,
                    offset: 10.,
                    period: 8,
                },
                WaveConfig {
                    wave: Waves::Pwm,
                    offset: 1.,
                    period: 12,
                },
            ],
        };

        let cfgs = serde_json::to_string_pretty(&cfg)?;

        let pb = PathBuf::from_str(path)?;
        fs::write(pb, &cfgs)?;

        Ok(())
    }
}
