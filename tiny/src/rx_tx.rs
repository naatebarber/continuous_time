use std::error::Error;

use serde::{Deserialize, Serialize};

use crate::{
    api::{InfFrame, OutputFrame, StepFrame},
    config::Config,
    network::Network,
};

pub fn load_socks(
    pull_from: &str,
    push_to: &str,
) -> Result<(zmq::Socket, zmq::Socket), Box<dyn Error>> {
    let context = zmq::Context::new();
    let pull_sock = context.socket(zmq::PULL)?;
    let push_sock = context.socket(zmq::PUSH)?;

    pull_sock.bind(pull_from)?;
    push_sock.bind(push_to)?;

    Ok((pull_sock, push_sock))
}

pub fn unpack_pull_sock(socket: &zmq::Socket) -> Result<(String, Vec<u8>), Box<dyn Error>> {
    let msgb = socket.recv_multipart(0)?;

    if msgb.len() < 2 {
        return Err("Invalid message length".into());
    }

    let cmd = String::from_utf8(msgb[0].to_vec())?;
    let data = msgb[1].to_vec();

    Ok((cmd, data))
}

pub fn pack_push_sock<'a, T>(socket: &zmq::Socket, cmd: &str, data: T) -> Result<(), Box<dyn Error>>
where
    T: Serialize + Deserialize<'a>,
{
    let msgb = serde_json::to_vec(&data)?;
    socket.send_multipart([cmd.as_bytes(), &msgb], zmq::DONTWAIT)?;

    Ok(())
}

pub fn go(config: Config, mut model: Network) -> Result<(), Box<dyn Error>> {
    let (pull_sock, push_sock) = load_socks(&config.pull_from, &config.push_to)?;

    let mut socks = [pull_sock.as_poll_item(zmq::POLLIN)];

    loop {
        match zmq::poll(&mut socks, 10) {
            Err(e) => {
                eprintln!("(tiny engine) failed to poll sockets: {}", e)
            }
            _ => (),
        }

        if socks[0].is_readable() {
            let (cmd, data) = match unpack_pull_sock(&pull_sock) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("(tiny engine) failed to unpack pull socket: {}", e);
                    continue;
                }
            };

            match cmd.as_str() {
                "step" => {
                    let step_frame: StepFrame = match serde_json::from_slice(&data) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("(tiny engine) failed to deserialize step_frame: {}", e);
                            continue;
                        }
                    };

                    let StepFrame {
                        inputs,
                        tau,
                        targets,
                    } = step_frame;

                    let (outputs, loss) = match model.step(
                        inputs,
                        tau,
                        config.steps,
                        targets,
                        config.retain,
                        config.learning_rate,
                    ) {
                        Some(state) => state,
                        None => {
                            println!("(tiny engine) no step state - network not ready");
                            continue;
                        }
                    };

                    let out_frame = OutputFrame {
                        outputs,
                        tau: model.tau,
                        loss,
                    };

                    match pack_push_sock(&push_sock, "out", out_frame) {
                        Err(e) => {
                            eprintln!("(tiny engine) failed to send out_frame: {}", e);
                            continue;
                        }
                        _ => (),
                    }
                }
                "inference" => {
                    let inf_frame: InfFrame = match serde_json::from_slice(&data) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("(tiny engine) failed to deserialize inf_frame: {}", e);
                            continue;
                        }
                    };

                    let InfFrame { inputs, tau } = inf_frame;

                    let outputs = match model.forward(inputs, tau, config.steps, None) {
                        Some(state) => state,
                        None => {
                            println!("(tiny engine) no forward state - network not ready");
                            continue;
                        }
                    };

                    let out_frame = OutputFrame {
                        outputs,
                        tau: model.tau,
                        loss: 0.,
                    };

                    match pack_push_sock(&push_sock, "out", out_frame) {
                        Err(e) => {
                            eprintln!("(tiny engine) failed to send out_frame: {}", e);
                            continue;
                        }
                        _ => (),
                    }
                }
                "drain" => {
                    println!("(tiny engine) received drain command: TODO hahah")
                }
                _ => (),
            }
        }
    }
}
