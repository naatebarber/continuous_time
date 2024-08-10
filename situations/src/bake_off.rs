use rand::{thread_rng, Rng};
use tiny::ContinuousNetwork;

use crate::util::get_ts;

pub fn bake_off(mut a: impl ContinuousNetwork, mut b: impl ContinuousNetwork) {
    let mut rng = thread_rng();

    let mut a_completed_steps = 0;
    let mut b_completed_steps = 0;

    let bake_time = 5.;
    let log_interval = 0.5;
    let mut last_log = get_ts();

    let start_a = get_ts();
    let mut a_tau = 0.0;

    while start_a + bake_time > get_ts() {
        let random_x = rng.gen::<f64>();
        let random_y = rng.gen::<f64>();

        a.step(vec![random_x], a_tau, 50, vec![random_y], 1000, 0.001);

        a_completed_steps += 1;
        a_tau += 0.1;

        if get_ts() - last_log > log_interval {
            println!("baking a...");
            last_log = get_ts();
        }
    }

    let start_b = get_ts();
    let mut b_tau = 0.0;

    while start_b + bake_time > get_ts() {
        let random_x = rng.gen::<f64>();
        let random_y = rng.gen::<f64>();

        b.step(vec![random_x], b_tau, 50, vec![random_y], 1000, 0.001);

        b_completed_steps += 1;
        b_tau += 0.1;

        if get_ts() - last_log > log_interval {
            println!("baking b...");
            last_log = get_ts();
        }
    }

    println!("a) {} steps in {} seconds", a_completed_steps, bake_time);
    println!("b) {} steps in {} seconds", b_completed_steps, bake_time);

    let (min, max) = match a_completed_steps > b_completed_steps {
        true => (b_completed_steps, a_completed_steps),
        false => (a_completed_steps, b_completed_steps),
    };

    let perc_diff = ((max as f64 / min as f64) * 100.) as isize;

    match a_completed_steps > b_completed_steps {
        true => println!("a is {}% more powerful than b", perc_diff),
        false => println!("b is {}% more powerful than a", perc_diff),
    }
}
