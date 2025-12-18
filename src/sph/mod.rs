use std::ops::{Add, AddAssign, Index, Mul};

use crate::const_value::{FRAC_1_SQRT_2PI, SQRT_2_POW_N_MOD_2_DIV_N, SQRT_GAMMA_N_DIV_2_EXP, SQRT_GAMMA_N_DIV_2_FRAC};
use crate::utilies::*;
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
use faer::linalg::solvers::{Eigen, SelfAdjointEigen};
use num_complex::Complex;
mod band_solve;
use crate::sp_func;

//__________________________________________________________________________
#[cfg(test)]
fn sph_norm_sqr(n: f64, a: f64, b: f64) -> f64 {
    use std::f64::consts::LN_2;
    ((LN_2 * (a + b + 1.0) + sp_func::real_lgamma(n + a + 1.0) + sp_func::real_lgamma(n + b + 1.0) - sp_func::real_lgamma(n + a + b + 1.0) - sp_func::real_lgamma(n + 1.0)).exp() / (2.0 * n + a + b + 1.0)).sqrt()
}
#[derive(Debug)]
#[repr(C)]
pub struct JacNorm {
    a: f64,
    b: f64,
    norm: Vec<f64>,
}
impl Index<usize> for JacNorm {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.norm[index]
    }
}
impl JacNorm {
    #[inline]
    fn new(a: f64, b: f64) -> JacNorm {
        JacNorm { a, b, norm: Vec::new() }
    }
    #[inline]
    fn set(&mut self, n: usize) {
        //2a \in \mathbb{N}
        //2b \in \mathbb{N}
        //a+b\in \mathbb{N}
        let a2 = (2.0 * self.a) as usize;
        let b2 = (2.0 * self.b) as usize;
        let apb = (self.a + self.b) as usize;

        for n in self.norm.len()..=n {
            //((a+b+1).exp2()/(2*n+a+b+1)).sqrt()*(gamma(n+a+1)*gamma(n+a+1)/gamma(n+a+b+1)/gamma(n+1)).sqrt()
            //=2.0_f64.powi((a+b+1)/2)*(((2*n+a+b+1)%2).exp2()/(2*n+a+b+1)).sqrt()*(gamma((2*n+2*a+2)/2)*gamma((2*n+2*b+2)/2)/gamma((2*n+2*a+2*b+2)/2)/gamma((2*n+2)/2)).sqrt()
            //=2.0_f64.powi((a+b+1)/2)*SQRT_2_POW_N_MOD_2_DIV_N[n]*(SQRT_GAMMA_N_DIV_2[2*n+2*a+2]*SQRT_GAMMA_N_DIV_2[2*n+2*b+2]/SQRT_GAMMA_N_DIV_2[2*n+2*a+2*b+2]/SQRT_GAMMA_N_DIV_2[2*n+2])*2.0_f64.powi(SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2*a+2]+SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2*b+2]-SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2*a+2*b+2]-SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2])
            //=SQRT_2_POW_N_MOD_2_DIV_N[2*n+a+b+1]*(SQRT_GAMMA_N_DIV_2[2*n+2*a+2]*SQRT_GAMMA_N_DIV_2[2*n+2*b+2]/SQRT_GAMMA_N_DIV_2[2*n+2*a+2*b+2]/SQRT_GAMMA_N_DIV_2[2*n+2])*2.0_f64.powi((a+b+1)/2+SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2*a+2]+SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2*b+2]-SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2*a+2*b+2]-SQRT_GAMMA_N_DIV_2_EXP_PART[2*n+2])

            let norm = SQRT_2_POW_N_MOD_2_DIV_N[2 * n + apb + 1] * (SQRT_GAMMA_N_DIV_2_FRAC[2 * n + a2 + 2] * SQRT_GAMMA_N_DIV_2_FRAC[2 * n + b2 + 2] / SQRT_GAMMA_N_DIV_2_FRAC[2 * n + a2 + b2 + 2] / SQRT_GAMMA_N_DIV_2_FRAC[2 * n + 2]) * 2.0_f64.powi(apb.div_ceil(2) as i32 + SQRT_GAMMA_N_DIV_2_EXP[2 * n + a2 + 2] + SQRT_GAMMA_N_DIV_2_EXP[2 * n + b2 + 2] - SQRT_GAMMA_N_DIV_2_EXP[2 * n + a2 + b2 + 2] - SQRT_GAMMA_N_DIV_2_EXP[2 * n + 2]);
            self.norm.push(norm);
        }
    }
    #[inline]
    fn tail(&self, x: f64) -> Res<f64> {
        //(1.0 - x).powf(self.a / 2.0) * (1.0 + x).powf(self.b / 2.0)
        //=(1.0 - x).powi(self.a * 2.0) * (1.0 + x).powi(self.b * 2.0).sqrt().sqrt()

        let tail = ((1.0 - x).powi((self.a * 2.0) as i32) * (1.0 + x).powi((self.b * 2.0) as i32)).sqrt().sqrt() * FRAC_1_SQRT_2PI;
        let dtail = tail * (self.a / (-1. + x) + self.b / (1. + x)) / 2.;
        Res(tail, dtail)
    }
    #[inline]
    fn jacobi_poly_n01(&self, x: f64) -> (Res<f64>, Res<f64>) {
        (Res(1.0, 0.0), Res((self.a + 1.0) + (self.a + self.b + 2.0) * (x - 1.0) / 2.0, (self.a + self.b + 2.0) / 2.0))
    }
    #[inline]
    fn jacobi_poly_n(&self, x: f64, n: usize, jn_1: Res<f64>, jn_2: Res<f64>) -> Res<f64> {
        let jn = sp_func::jacobi_poly_n(n as f64, self.a, self.b, x, jn_1.0, jn_2.0);
        let djn = sp_func::djacobi_poly_n(n as f64, self.a, self.b, x, jn, jn_1.0);
        Res(jn, djn)
    }
}
#[test]
fn norm_check() {
    let a = 20.5;
    let b = 3.5;
    let mut jac = JacNorm::new(a, b);
    jac.set(60);
    let error = sph_norm_sqr(60.0, a, b) - jac[60];
    assert!(error < 1e-9)
}
#[derive(Debug)]
#[repr(C)]
pub struct Spheroidal<T>
// where
//     T: Zero,
{
    s: f64,
    m: f64,
    c: T,
    pub lambda: T,
    pub spheroical_coe: Vec<T>,

    jac_norm: JacNorm,
}
impl<T:FC64> Spheroidal<T>{
    pub fn lambdA(&self)->T{
        self.lambda - self.c * 2.0 * self.m + self.c * self.c
    }
}
pub trait InnerRequire<T> {
    fn coe(&self) -> &Vec<T>;
    fn jac(&self) -> &JacNorm;
}
pub trait SphEigen<T>
where
    T: Zero + Add<f64, Output = T> + Mul<f64, Output = T> + Copy,
    Res<f64>: Mul<T, Output = Res<T>> + Mul<Res<T>, Output = Res<T>>,
    Res<T>: AddAssign,
    Self: InnerRequire<T>,
{
    fn new_raw(s: f64, l: f64, m: f64, c: T, delta: f64) -> Spheroidal<T>;
    fn new(s: f64, l: f64, m: f64, c: T, delta: f64) -> Spheroidal<T> {
        let mut sph = Self::new_raw(s, l, m, c, delta);
        let len = sph.spheroical_coe.len();
        sph.jac_norm.set(len);
        sph
    }

    fn at(&self, x: f64) -> Res<T> {
        let nmax = self.coe().len();
        let tail = self.jac().tail(x);

        let (mut jn0, mut jn1) = self.jac().jacobi_poly_n01(x);
        let mut ans = Res::<T>::zero();
        ans += (jn0 / self.jac()[0]) * self.coe()[0];
        ans += (jn1 / self.jac()[1]) * self.coe()[1];
        for n in 2..nmax {
            let jn2 = self.jac().jacobi_poly_n(x, n, jn1, jn0);
            ans += (jn2 / self.jac()[n]) * self.coe()[n];
            // println!("{}",jn2);
            jn0 = jn1;
            jn1 = jn2;
        }
        tail * ans
    }
}
impl<T> InnerRequire<T> for Spheroidal<T>
where
    T: Zero,
{
    #[inline]
    fn coe(&self) -> &Vec<T> {
        &self.spheroical_coe
    }
    #[inline]
    fn jac(&self) -> &JacNorm {
        &self.jac_norm
    }
}
impl SphEigen<f64> for Spheroidal<f64> {
    fn new_raw(s: f64, l: f64, m: f64, c: f64, delta: f64) -> Spheroidal<f64> {
        let a = (m + s).abs();
        let b = (m - s).abs();

        let l_ini = m.abs().max(s.abs());
        if l < l_ini {
            panic!("l can't smaller than floor(max(|m|,|s|)),l={},m={},s={}", l as i32, m as i32, s as i32)
        }
        let (mut m2, mut m1, mut m0) = band_solve::spec_matrix_upper_ini(l_ini, l, m, s, c);

        let position = (l - l_ini) as usize;
        #[cfg(any(feature = "mkl", feature = "openblas"))]
        let (_r, e, v) = loop {
            let (r, e, v) = band_solve::spec_ev_real((&m2, &m1, &m0));

            if (v[position + (m0.len() - 1) * m0.len()]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), l_ini, m, s, c);
            } else {
                break (r, e, v);
            }
        };
        #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
        let eigen = loop {
            let eigen = band_solve::spec_ev_real((&m2, &m1, &m0));

            if (eigen.U()[(position, (m0.len() - 1))]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), l_ini, m, s, c);
            } else {
                break eigen;
            }
        };
        let mut coe: Vec<f64> = Vec::new();
        for i in 0..m0.len() {
            #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
            coe.push(eigen.U()[(i, position)]);
            #[cfg(any(feature = "mkl", feature = "openblas"))]
            coe.push(v[position + i * m0.len()]);
        }
        #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
        let lambda = eigen.S()[position];
        #[cfg(any(feature = "mkl", feature = "openblas"))]
        let lambda = e[position];
        Spheroidal { s, m, c, lambda, spheroical_coe: coe, jac_norm: JacNorm::new(a, b) }
    }
}
impl SphEigen<Complex<f64>> for Spheroidal<Complex<f64>> {
    fn new_raw(s: f64, l: f64, m: f64, c: Complex<f64>, delta: f64) -> Spheroidal<Complex<f64>> {
        let a = (m + s).abs();
        let b = (m - s).abs();

        let l_ini = m.abs().max(s.abs());
        if l < l_ini {
            panic!("l can't smaller than floor(max(|m|,|s|)),l={},m={},s={}", l as i32, m as i32, s as i32)
        }
        let (mut m2, mut m1, mut m0) = band_solve::spec_matrix_upper_ini(l_ini, l, m, s, c);

        let position = (l - l_ini) as usize;
        #[cfg(any(feature = "mkl", feature = "openblas"))]
        let (_r, e, v) = loop {
            let (r, e, v) = band_solve::spec_ev_complex((&m2, &m1, &m0));

            let e_err = (v[position + (m0.len() - 1) * m0.len()]).abs();
            if e_err > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), l_ini, m, s, c);
            } else {
                break (r, e, v);
            }
        };
        #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
        let eigen = loop {
            let eigen = band_solve::spec_ev_complex((&m2, &m1, &m0));

            if (eigen.U()[(position, (m0.len() - 1))]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), l_ini, m, s, c);
            } else {
                break eigen;
            }
        };
        let mut coe: Vec<Complex<f64>> = Vec::new();
        for i in 0..m0.len() {
            #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
            coe.push(eigen.U()[(position, i)]);
            #[cfg(any(feature = "mkl", feature = "openblas"))]
            coe.push(v[position + i * m0.len()]);
        }
        #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
        let lambda = eigen.S()[position];
        #[cfg(any(feature = "mkl", feature = "openblas"))]
        let lambda = e[position];
        Spheroidal { s, m, c, lambda, spheroical_coe: coe, jac_norm: JacNorm::new(a, b) }
    }
}

//__________________________________________________________________________
#[derive(Debug)]
pub struct Spherical {
    x: f64,
    jac_norm: JacNorm,
    tailx: Res<f64>,
    poly: Vec<Res<f64>>,
}
impl Spherical {
    pub fn new(s: f64, m: f64, x: f64) -> Self {
        let a = (m + s).abs();
        let b = (m - s).abs();
        const N: usize = 20;
        let mut jac_norm = JacNorm::new(a, b);
        jac_norm.set(N);

        let tailx = jac_norm.tail(x);
        let (j0, j1) = jac_norm.jacobi_poly_n01(x);
        // if n==1{
        //     return Self {x,jac_norm,tailx,poly:vec![j0]};
        // }
        let mut sph = vec![j0, j1];
        for i in 2..N {
            let jn = jac_norm.jacobi_poly_n(x, i, sph[i - 1], sph[i - 2]);
            sph.push(jn);
        }
        Self { x, jac_norm, tailx, poly: sph }
    }
    fn set(&mut self, n: usize) {
        self.jac_norm.set(n);
        for i in self.poly.len()..n {
            let jn = self.jac_norm.jacobi_poly_n(self.x, i, self.poly[i - 1], self.poly[i - 2]);
            self.poly.push(jn);
        }
    }
}
#[derive(Debug)]
#[repr(C)]
pub struct SphCoe<T> {
    lmax: f64,
    lmin: f64,
    jac_norm: JacNorm,

    #[cfg(any(feature = "mkl", feature = "openblas"))]
    lambda: Vec<T>,
    #[cfg(any(feature = "mkl", feature = "openblas"))]
    spheroical_coe: Vec<T>,

    #[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
    eigen: T,
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
pub type SphCoeF64 = SphCoe<f64>;
#[cfg(any(feature = "mkl", feature = "openblas"))]
pub type SphCoeC64 = SphCoe<Complex<f64>>;
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
pub type SphCoeF64 = SphCoe<SelfAdjointEigen<f64>>;
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
pub type SphCoeC64 = SphCoe<Eigen<f64>>;
pub trait InnerRequireList<T> {
    fn len(&self) -> usize;
    fn lmin(&self) -> f64;
    fn lambda_index(&self, index: usize) -> T;
    fn coe_index(&self, index: (usize, usize)) -> T;
}
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
impl InnerRequireList<f64> for SphCoe<SelfAdjointEigen<f64>> {
    fn len(&self) -> usize {
        self.eigen.S().dim()
    }
    fn lmin(&self) -> f64 {
        self.lmin
    }
    fn lambda_index(&self, index: usize) -> f64 {
        self.eigen.S()[index]
    }
    fn coe_index(&self, index: (usize, usize)) -> f64 {
        self.eigen.U()[index]
    }
}
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
impl InnerRequireList<Complex<f64>> for SphCoe<Eigen<f64>> {
    fn len(&self) -> usize {
        self.eigen.S().dim()
    }
    fn lmin(&self) -> f64 {
        self.lmin
    }
    fn lambda_index(&self, index: usize) -> Complex<f64> {
        self.eigen.S()[index]
    }
    fn coe_index(&self, index: (usize, usize)) -> Complex<f64> {
        self.eigen.U()[index]
    }
}
pub trait SphListEigen<T>
where
    T: FC64,
    Res<f64>: Mul<T, Output = Res<T>> + Mul<Res<T>, Output = Res<T>>,
    Res<T>: AddAssign,
    Self: InnerRequireList<T>,
{
    fn new(s: f64, lmax: f64, m: f64, c: T, delta: f64) -> Self;

    fn at(&self, spherical_at_x: &mut Spherical, l: f64) -> Res<T> {
        let lmax_lmin = self.len(); //lmax-lmin
        let l_lmin = (l - self.lmin()) as usize; //l-lmin

        if l_lmin > lmax_lmin {
            unreachable!()
        }
        if spherical_at_x.poly.len() < self.len() {
            spherical_at_x.set(self.len());
        }
        let mut ans = Res::zero();
        for i in 0..lmax_lmin {
            if i > l_lmin && self.coe_index((i, l_lmin)).abs() < 1e-15 {
                break;
            }
            ans += spherical_at_x.poly[i] / spherical_at_x.jac_norm[i] * self.coe_index((i, l_lmin));
        }
        spherical_at_x.tailx * ans
    }
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
impl<T> InnerRequireList<T> for SphCoe<T>
where
    T: Zero + Add<f64, Output = T> + Mul<f64, Output = T> + Copy,
    Res<f64>: Mul<T, Output = Res<T>> + Mul<Res<T>, Output = Res<T>>,
    Res<T>: AddAssign,
{
    fn len(&self) -> usize {
        self.lambda.len()
    }
    fn coe_index(&self, index: (usize, usize)) -> T {
        self.spheroical_coe[index.0 * self.len() + index.1]
    }
    fn lambda_index(&self, index: usize) -> T {
        self.lambda[index]
    }
    fn lmin(&self) -> f64 {
        self.lmin
    }
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
impl SphListEigen<f64> for SphCoe<f64> {
    fn new(s: f64, lmax: f64, m: f64, c: f64, delta: f64) -> SphCoe<f64> {
        let a = (m + s).abs();
        let b = (m - s).abs();
        let lmin = m.abs().max(s.abs());
        if lmax < lmin {
            panic!("l can't smaller than max(|m|,|s|),l={},m={},s={}", lmax as i32, m as i32, s as i32)
        }
        let (mut m2, mut m1, mut m0) = band_solve::spec_matrix_upper_ini(lmin, lmax, m, s, c);

        let position = (lmax - lmin) as usize;
        let (_r, e, v) = loop {
            let (r, e, v) = band_solve::spec_ev_real((&m2, &m1, &m0));

            if (v[position + (m0.len() - 1) * m0.len()]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), lmin, m, s, c);
            } else {
                break (r, e, v);
            }
        };
        let mut jac = JacNorm::new(a, b);
        jac.set(e.len());
        SphCoe { lmax, lmin, lambda: e, spheroical_coe: v, jac_norm: jac }
    }
}
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
impl SphListEigen<f64> for SphCoe<SelfAdjointEigen<f64>> {
    fn new(s: f64, lmax: f64, m: f64, c: f64, delta: f64) -> Self {
        let a = (m + s).abs();
        let b = (m - s).abs();
        let lmin = m.abs().max(s.abs());
        if lmax < lmin {
            panic!("l can't smaller than max(|m|,|s|),l={},m={},s={}", lmax as i32, m as i32, s as i32)
        }
        let (mut m2, mut m1, mut m0) = band_solve::spec_matrix_upper_ini(lmin, lmax, m, s, c);

        let position = (lmax - lmin) as usize;
        let eigen = loop {
            let eigen = band_solve::spec_ev_real((&m2, &m1, &m0));

            if (eigen.U()[(position, (m0.len() - 1))]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), lmin, m, s, c);
            } else {
                break eigen;
            }
        };
        let mut jac = JacNorm::new(a, b);
        jac.set(m0.len());
        SphCoe { lmax, lmin, jac_norm: jac, eigen }
    }
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
impl SphListEigen<Complex<f64>> for SphCoe<Complex<f64>> {
    fn new(s: f64, lmax: f64, m: f64, c: Complex<f64>, delta: f64) -> SphCoe<Complex<f64>> {
        let a = (m + s).abs();
        let b = (m - s).abs();
        let lmin = m.abs().max(s.abs());
        if lmax < lmin {
            panic!("l can't smaller than max(|m|,|s|),l={},m={},s={}", lmax as i32, m as i32, s as i32)
        }
        let (mut m2, mut m1, mut m0) = band_solve::spec_matrix_upper_ini(lmin, lmax, m, s, c);

        let position = (lmax - lmin) as usize;
        let (_r, e, v) = loop {
            let (r, e, v) = band_solve::spec_ev_complex((&m2, &m1, &m0));

            if (v[position + (m0.len() - 1) * m0.len()]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), lmin, m, s, c);
            } else {
                break (r, e, v);
            }
        };
        let mut jac = JacNorm::new(a, b);
        jac.set(e.len());
        SphCoe { lmax, lmin, lambda: e, spheroical_coe: v, jac_norm: jac }
    }
}
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
impl SphListEigen<Complex<f64>> for SphCoe<Eigen<f64>> {
    fn new(s: f64, lmax: f64, m: f64, c: Complex<f64>, delta: f64) -> Self {
        let a = (m + s).abs();
        let b = (m - s).abs();
        let lmin = m.abs().max(s.abs());
        if lmax < lmin {
            panic!("l can't smaller than max(|m|,|s|),l={},m={},s={}", lmax as i32, m as i32, s as i32)
        }
        let (mut m2, mut m1, mut m0) = band_solve::spec_matrix_upper_ini(lmin, lmax, m, s, c);

        let position = (lmax - lmin) as usize;
        let eigen = loop {
            let eigen = band_solve::spec_ev_complex((&m2, &m1, &m0));

            if (eigen.U()[(position, (m0.len() - 1))]).abs() > delta {
                band_solve::spec_matrix_extend5((&mut m2, &mut m1, &mut m0), lmin, m, s, c);
            } else {
                break eigen;
            }
        };
        let mut jac = JacNorm::new(a, b);
        jac.set(m0.len());
        SphCoe { lmax, lmin, jac_norm: jac, eigen }
    }
}
#[test]
fn sphlisttest() {
    let s = -2.0;
    let l = 12.0;
    let m = 2.0;
    let c = -6.5;

    let sphlist = SphCoe::new(s, 20.0, 2.0, c, 1e-15);
    let mut spherical_at_x = Spherical::new(s, m, 0.37);
    let sph = Spheroidal::new(s, l, m, c, 1e-15);

    println!("{:?}", sphlist.at(&mut spherical_at_x, 12.0));
    println!("{:?}", sph.at(0.37));
}
#[test]
fn test_qnm() {
    let l = 27.0;
    let m = 2.0;
    let s = -2.0;
    let c: Complex<f64> = 0.5 * (3.4922780330834677 - 0.0938655029824283 * Complex::i());
    let a = Spheroidal::new(s, l, m, c, 1e-12);
    print!("{}", a.at(0.23))
    //-0.027645889810103078, im: -0.0011723520291766199
}
#[test]
fn size_check() {
    use std::mem::{align_of, size_of};

    println!("size of f64 = {}", size_of::<f64>());
    println!("size of Spheroidal<f64> = {}", size_of::<Spheroidal<f64>>());
    println!("align of Spheroidal<f64> = {}", align_of::<Spheroidal<f64>>());
    println!("size of Spheroidal<Complex<f64>> = {}", size_of::<Spheroidal<Complex<f64>>>());
    println!("align of Spheroidal<Complex<f64>> = {}", align_of::<Spheroidal<Complex<f64>>>());

    // println!("size of SpheroidalList<f64> = {}",size_of::<SpheroidalList<f64>>());
    // println!("align of SpheroidalList<f64> = {}",align_of::<SpheroidalList<f64>>());
    // println!("size of SpheroidalList<Complex<f64>> = {}",size_of::<SpheroidalList<Complex<f64>>>());
    // println!("align of SpheroidalList<Complex<f64>> = {}",align_of::<SpheroidalList<Complex<f64>>>());
}
