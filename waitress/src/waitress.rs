use std::io::{stdout, Write};
use std::{collections::HashMap, error::Error, thread, time::Duration};

use rand::{prelude::*, thread_rng};
use serde::{Deserialize, Serialize};

use crate::util::fuzz;
use crate::{config::WaitressConfig, util::get_ts};

pub type Wave = Box<dyn Fn(f64, &mut ThreadRng, &WaitressConfig) -> f64>;

pub struct Waitress {
    pub pub_sock: zmq::Socket,
    pub config: WaitressConfig,
    pub base: f64,
    pub waves: Vec<Wave>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Frame {
    ts: f64,
    channels: HashMap<String, f64>,
}

impl Waitress {
    pub fn new(config: WaitressConfig) -> Result<Waitress, Box<dyn Error>> {
        let context = zmq::Context::new();
        let pub_sock = context.socket(zmq::PUB)?;
        pub_sock.bind(&config.pub_to)?;

        Ok(Waitress {
            pub_sock,
            config,
            base: 0.,
            waves: vec![],
        })
    }

    pub fn wave_sinusoidal(&mut self, offset: f64) {
        self.waves.push(Box::new(
            move |v: f64, rng: &mut ThreadRng, config: &WaitressConfig| -> f64 {
                let pure_v = f64::sin(v + offset) * config.maxv + config.minv;

                fuzz(pure_v, config.noise, rng)
            },
        ));
    }

    pub fn wave_periodic_spike(&mut self, offset: f64, period: usize) {
        self.waves.push(Box::new(
            move |v: f64, rng: &mut ThreadRng, config: &WaitressConfig| -> f64 {
                let pure_v = if (v + offset).floor() as usize % period == 0 {
                    config.maxv
                } else {
                    config.minv
                };

                fuzz(pure_v, config.noise, rng)
            },
        ));
    }

    pub fn wave_periodic_spike_decay(&mut self, offset: f64, period: usize) {
        self.waves.push(Box::new(
            move |v: f64, rng: &mut ThreadRng, config: &WaitressConfig| -> f64 {
                let pure_v = if (v + offset).floor() as usize % period == 0 {
                    config.maxv
                } else {
                    config.minv
                };

                fuzz(pure_v, config.noise, rng)
            },
        ))
    }

    pub fn wave_pwm(&mut self, offset: f64, period: usize) {
        self.waves.push(Box::new(
            move |v: f64, rng: &mut ThreadRng, config: &WaitressConfig| -> f64 {
                let pure_v = if ((v + offset) / period as f64).floor() as usize % 2 == 0 {
                    config.maxv
                } else {
                    config.minv
                };

                fuzz(pure_v, config.noise, rng)
            },
        ))
    }

    pub fn send_frame(&self, frame_v: f64) -> Result<(), Box<dyn Error>> {
        let cur_ts = get_ts();

        let mut channels = HashMap::new();
        channels.insert(self.config.channel.clone(), frame_v);

        let frame = Frame {
            ts: cur_ts,
            channels,
        };

        let frame_vec = serde_json::to_vec(&frame)?;

        match self.pub_sock.send_multipart(
            ["waitress".as_bytes(), "feed".as_bytes(), &frame_vec],
            zmq::DONTWAIT,
        ) {
            Ok(_) => print!("."),
            Err(_) => print!("x"),
        };

        let _ = stdout().flush();

        Ok(())
    }

    pub fn go(&mut self) {
        let mut rng = thread_rng();

        loop {
            let mut frame_v = 0.;

            self.waves.iter().for_each(|wave| {
                let wave_v = wave(self.base, &mut rng, &self.config);
                frame_v += wave_v;
            });

            frame_v += self.config.offsetv;

            match self.send_frame(frame_v) {
                Err(e) => eprintln!("waitress failed to send frame: {}", e),
                _ => (),
            }

            self.base += self.config.step;

            thread::sleep(Duration::from_secs_f64(self.config.sleep));
        }
    }
}
