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
    let mut a = tiny::SsmNetwork::new(200, 1, 1);
    a.weave(1.);

    let mut b = tiny::PowerNetwork::new(200, 1, 1, 12).unwrap();
    b.weave(1.);

    let mut z = tiny::HashNetwork::new(200, 1, 1);
    z.weave(1.);

    let steps = 50;
    let retain = 8000;
    // bake_off(b, z);

    churn_one(b, steps, retain, 5.);
}

fn main() {
    perf()
}
