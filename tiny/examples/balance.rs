use std::f64::consts;
use tiny::get_ts;

const G: f64 = 9.8;

pub struct InvertedPendulum {
    pub tau: f64,
    pub mass: f64,
    pub theta: f64,
    pub a: f64,
    pub v: f64,
    pub length: f64, // Length of the rod
}

impl InvertedPendulum {
    pub fn new(i_theta: f64, i_a: f64, i_v: f64, length: f64) -> InvertedPendulum {
        InvertedPendulum {
            tau: get_ts(),
            theta: i_theta,
            mass: 10.,
            a: i_a,
            v: i_v,
            length, // Initialize the length of the rod
        }
    }

    pub fn observation(&self) -> (f64, f64, f64) {
        return (self.theta, self.a, self.v);
    }

    pub fn step(&mut self, step_size: f64, thrust_ccw: f64, thrust_cw: f64) -> (f64, f64) {
        // Calculate the net thrust force (clockwise is positive, counterclockwise is negative)
        let net_thrust = thrust_cw.abs() - thrust_ccw.abs();

        // Calculate the torque due to gravity: τ_gravity = -m * g * L * sin(θ)
        let torque_gravity = -self.mass * G * self.length * self.theta.sin();

        // Calculate the torque due to the thrusters: τ_thrust = L * net_thrust
        let torque_thrust = self.length * net_thrust;

        // Calculate the total torque: τ_total = τ_gravity + τ_thrust
        let torque_total = torque_gravity + torque_thrust;

        // Update angular acceleration: α = τ_total / I, where I = m * L^2
        let moment_of_inertia = self.mass * self.length.powi(2);
        self.a = torque_total / moment_of_inertia;

        // Update angular velocity: v += α * Δt
        self.v += self.a * step_size;
        self.v *= 0.99;

        // Update angle: θ += v * Δt
        self.theta += self.v * step_size;

        return self.loss(self.theta, thrust_cw, thrust_ccw);
    }

    pub fn loss(&self, theta: f64, tccw: f64, tcw: f64) -> (f64, f64) {
        let theta_norm = ((2. * consts::PI) % theta).abs();
        let theta_loss = consts::PI - theta_norm;

        let sum_thrust = tccw + tcw;

        if theta > 0. && theta < consts::PI {
            // should be thrusting clockwise
            return (
                1. - (tcw / sum_thrust) + theta_loss,
                tccw / sum_thrust + theta_loss,
            );
        } else {
            return (
                1. - (tccw / sum_thrust) + theta_loss,
                tcw / sum_thrust + theta_loss,
            );
        }
    }
}

pub fn main() {}
