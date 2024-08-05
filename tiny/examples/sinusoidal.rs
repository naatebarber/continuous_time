use core::f64;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tiny::Network;

pub fn get_ts() -> f64 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_secs_f64()
}

fn time_dependent_sin(ts: f64) -> f64 {
    let window = 3.;
    let modulus = ts % window;
    let ratio = (modulus / window) * f64::consts::PI * 2.;
    return 20. * f64::sin(ratio);
}

fn main() {
    let mut network = Network::new(12, 1, 1);
    network.weave(0.5);

    let learning_rate = 0.004;
    let steps = 100;
    let retain = 100;

    let interval = 0.1;

    loop {
        let ts = get_ts();
        let input = time_dependent_sin(ts);
        let target_tau = ts + interval;
        let target = time_dependent_sin(target_tau);

        let output = network.step(
            vec![input],
            ts + interval,
            steps,
            vec![target],
            retain,
            learning_rate,
        );

        if let Some(output) = output {
            // println!("(tiny example) output {:?} target {:?}", output.0, vec![target]);
            println!(
                "(tiny example) loss {:?} input {:?} output {:?}",
                output.1, input, output.0[0]
            );
        }

        if get_ts() < target_tau {
            thread::sleep(Duration::from_secs_f64(target_tau - get_ts()))
        } else {
            println!("(tiny example) simulation behind schedule!")
        }
    }
}
