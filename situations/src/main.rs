use situations::{bake_off, churn_one};
use tiny::ContinuousNetwork;

pub fn compare() {
    println!("Hello, situation!");

    println!("creating hash network");
    let mut a = tiny::HashNetwork::new(24, 1, 1);
    a.weave(1.0);
    println!("creating ssm network");
    let mut b = tiny::SsmNetwork::new(24, 1, 1);
    b.weave(1.0);

    println!("starting bake off");
    bake_off(a, b);
}

pub fn perf() {
    let mut a = tiny::HashNetwork::new(24, 1, 1);
    a.weave(1.);

    churn_one(a);
}

fn main() {
    perf()
}
