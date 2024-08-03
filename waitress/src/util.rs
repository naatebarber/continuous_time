use rand::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn get_ts() -> f64 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_secs_f64()
}

pub fn fuzz(x: f64, noise: f64, rng: &mut ThreadRng) -> f64 {
    if noise - noise == 0. {
        return x;
    }

    let fuzz_mult = rng.gen_range(-noise..noise);
    x + (fuzz_mult * x)
}
