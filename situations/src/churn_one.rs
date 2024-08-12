use rand::{thread_rng, Rng};
use tiny::ContinuousNetwork;

use crate::util::get_ts;

pub fn churn_one(mut a: impl ContinuousNetwork, steps: usize, retain: usize, t: f64) {
    let mut rng = thread_rng();

    let mut a_completed_steps = 0;

    let log_interval = 0.5;
    let mut last_log = get_ts();

    let start_a = get_ts();
    let mut a_tau = 0.0;

    while start_a + t > get_ts() {
        let random_x = rng.gen::<f64>();
        let random_y = rng.gen::<f64>();

        a.step(vec![random_x], a_tau, steps, vec![random_y], retain, 0.001);

        a_completed_steps += 1;
        a_tau += 0.1;

        if get_ts() - last_log > log_interval {
            println!("baking a...");
            last_log = get_ts();
        }
    }

    let sps = a_completed_steps as f64 / (get_ts() - start_a);

    println!("{} steps/sec", sps);
}
