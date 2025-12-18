//RUSTFLAGS="-C target-cpu=native" cargo run -r --example error_plotter
use plotters::{prelude::*, style::full_palette::{ORANGE}};
extern crate teukolsky_ode_solver;
use num_complex::{Complex64, ComplexFloat};
use teukolsky_ode_solver::{jh,sph::{self, SphEigen}};
struct Sol<T>{
    s:f64,
    m:f64,
    a:f64,
    omega:T,
    lambda:T,
    _sph:sph::Spheroidal<T>,
    radial:jh::SE<T>
}
impl Sol<f64> {
    fn new(s:f64,l:f64,m:f64,a:f64,omega:f64)->Self{
        let c=a*omega;
        let sph=sph::Spheroidal::new(s, l, m, c, 1e-15);
        let lambda=sph.lambda-(2.*m*c-c*c);
        Sol { s,m,a,omega,lambda,_sph: sph, radial: jh::SE::new(s, l, m, a, omega, lambda) }
    }
    fn error(&self,r:f64)->f64{
        let ans=self.radial.cal3_at(r);
        let delta=r*r+self.a*self.a-2.*r;
        let k=(r*r+self.a*self.a)*self.omega-self.m*self.a;
        let v=self.lambda-4.*Complex64::I*self.s*self.omega*r-(k*k-2.*Complex64::I*self.s*(r-1.)*k)/delta;
        let err_in=ans.0[2]+(2.*(self.s+1.)*(r-1.)*ans.0[1]-v*ans.0[0])/delta;
        let err_up=ans.1[2]+(2.*(self.s+1.)*(r-1.)*ans.1[1]-v*ans.1[0])/delta;

        let rinmax=(2.*(self.s+1.)*(r-1.)*ans.0[1]/delta).abs().max((v*ans.0[0]/delta).abs()).max(ans.0[2].abs());
        let rupmax=(2.*(self.s+1.)*(r-1.)*ans.1[1]/delta).abs().max((v*ans.1[0]/delta).abs()).max(ans.1[2].abs());

        (err_in.abs()/rinmax).max(err_up.abs()/rupmax)
    }
}
// impl Sol<Complex64> {
//     fn new(s:f64,l:f64,m:f64,a:f64,omega:Complex64)->Self{
//         let c=a*omega;
//         let sph=sph::Spheroidal::new(s, l, m, c, 1e-15);
//         let lambda=sph.lambda-(2.*m*c-c*c);
//         Sol { s,m,a,omega,lambda,_sph: sph, radial: jh::SE::new(s, l, m, a, omega, lambda) }
//     }
//     fn error(&self,r:f64)->f64{
//         let ans=self.radial.cal3_at(r);
//         let delta=r*r+self.a*self.a-2.*r;
//         let k=(r*r+self.a*self.a)*self.omega-self.m*self.a;
//         let v=self.lambda-4.*Complex64::I*self.s*self.omega*r-(k*k-2.*Complex64::I*self.s*(r-1.)*k)/delta;
//         let err_in=ans.0[2]+(2.*(self.s+1.)*(r-1.)*ans.0[1]-v*ans.0[0])/delta;
//         let err_up=ans.1[2]+(2.*(self.s+1.)*(r-1.)*ans.1[1]-v*ans.1[0])/delta;

//         let rinmax=(2.*(self.s+1.)*(r-1.)*ans.0[1]/delta).abs().max((v*ans.0[0]/delta).abs()).max(ans.0[2].abs());
//         let rupmax=(2.*(self.s+1.)*(r-1.)*ans.1[1]/delta).abs().max((v*ans.1[0]/delta).abs()).max(ans.1[2].abs());

//         (err_in.abs()/rinmax).max(err_up.abs()/rupmax)
//     }
// }
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root=SVGBackend::new("examples\\epsilon.svg", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("error", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..200f64, (1e-16f64..1e-9f64).log_scale())?;

    chart.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

    let sol1: Sol<f64>=Sol::<f64>::new(-2.0, 2.0, 2.0, 0.99, 1e-9);
    chart
        .draw_series(LineSeries::new(
            (0..=800).map(|x| (x as f64 / 800.0)*198.0+2.0).map(|x| (x, sol1.error(x))),
            &RED,
        ))?
        .label("omega = 1e-9")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    let sol2: Sol<f64>=Sol::<f64>::new(-2.0, 2.0, 2.0, 0.99, 1.5);
    chart
        .draw_series(LineSeries::new(
            (0..=800).map(|x| (x as f64 / 800.0)*198.0+2.0).map(|x| (x, sol2.error(x))),
            &BLUE,
        ))?
        .label("omega = 1.5")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // let sol3: Sol<Complex64>=Sol::<Complex64>::new(-2.0, 2.0, 2.0, 0.99, 1.0+Complex64::I);
    // chart
    //     .draw_series(LineSeries::new(
    //         (0..=800).map(|x| (x as f64 / 800.0)*198.0+2.0).map(|x| (x, sol3.error(x))),
    //         &GREEN,
    //     ))?
    //     .label("omega = 1.0+i")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    let sol4: Sol<f64>=Sol::<f64>::new(-2.0, 2.0, 2.0, 0.99, 10.0);
    chart
        .draw_series(LineSeries::new(
            (0..=800).map(|x| (x as f64 / 800.0)*198.0+2.0).map(|x| (x, sol4.error(x))),
            &ORANGE,
        ))?
        .label("omega = 10.0")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE));
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}