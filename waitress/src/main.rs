extern crate getopts;
mod config;
mod util;
mod waitress;

use config::{WaitressConfig, Waves};
use getopts::Options;
use std::{env, process};
use waitress::Waitress;

type Opts = String;

fn parse_args() -> Opts {
    fn print_usage(program: &str, opts: Options) {
        let brief = format!("Usage: {} FILE [options]", program);
        print!("{}", opts.usage(&brief));
    }

    let args = env::args().collect::<Vec<String>>();
    let program = args[0].clone();
    let mut opts = Options::new();

    opts.optopt(
        "f",
        "file",
        "Waitress configuration file",
        "/path/to/waitress.cfg.json",
    );
    opts.optflag(
        "g",
        "generate-config",
        "Generate a config at your current path.",
    );

    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}", e);
            print_usage(&program, opts);
            process::exit(1);
        }
    };

    if matches.opt_present("g") {
        match WaitressConfig::dump("./waitress.cfg.json") {
            Ok(c) => c,
            Err(e) => {
                eprintln!("failed to write config: {}", e);
                process::exit(1)
            }
        };

        process::exit(0);
    }

    let config_path = matches.opt_str("f");

    if let Some(config_path) = config_path {
        return config_path;
    } else {
        print_usage(&program, opts);
        process::exit(0)
    }
}
fn main() {
    let cfgp = parse_args();

    let config = match WaitressConfig::load(&cfgp) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load config: {}", e);
            process::exit(1)
        }
    };

    let wave_definitions = config.waves.clone();
    let channel = config.channel.clone();

    let mut waitress = match Waitress::new(config) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("failed to create waitress: {}", e);
            process::exit(1);
        }
    };

    for wave_def in wave_definitions {
        println!(
            "(waitress:{}) streaming {:?} -> offset {} period {}",
            channel, wave_def.wave, wave_def.offset, wave_def.period
        );
        match wave_def.wave {
            Waves::Sinusoidal => waitress.wave_sinusoidal(wave_def.offset),
            Waves::Spike => waitress.wave_periodic_spike(wave_def.offset, wave_def.period),
            Waves::SpikeDecay => {
                waitress.wave_periodic_spike_decay(wave_def.offset, wave_def.period)
            }
            Waves::Pwm => waitress.wave_pwm(wave_def.offset, wave_def.period),
        }
    }

    waitress.go()
}
