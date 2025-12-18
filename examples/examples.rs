//RUSTFLAGS="-C target-cpu=native" cargo run -r --example examples
extern crate teukolsky_ode_solver;
use fastrand;
use std::hint::black_box;
use teukolsky_ode_solver::jh::SE;
use teukolsky_ode_solver::sph::{SphEigen, Spheroidal};
fn main() {

    let s = -2.0;
    let l = 20.0;
    let m = 2.0;
    let a = 0.9;

    for _ in 0..100000 {
        let omega = fastrand::f64();
        let c = a * omega;
        let sph = Spheroidal::new(s, l, m, c, 1e-15);
        let lambda = sph.lambdA();
        black_box(SE::new(s, l, m, a, omega, lambda));
    }
}
