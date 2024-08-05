extern crate getopts;
use getopts::Options;
use std::{env, process};

pub use tiny::config::Config;
pub use tiny::rx_tx;
use tiny::HashNetwork;

fn parse_args() -> String {
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
        let config = Config::new();
        match config.dump("./tiny.cfg.json") {
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
    let config_path = parse_args();

    let config = match Config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("(tiny engine) failed to load config: {}", e);
            process::exit(1)
        }
    };

    let mut network = HashNetwork::new(config.size, config.d_in, config.d_out);

    network.weave(config.density);

    match rx_tx::go(config, network) {
        Err(e) => eprintln!("(tiny engine) failed with error: {}", e),
        _ => (),
    }
}
