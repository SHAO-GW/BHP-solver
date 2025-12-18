#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
use faer::linalg::solvers::{Eigen, SelfAdjointEigen};
use num_complex::Complex;
#[cfg(any(feature = "mkl", feature = "openblas"))]
use std::{
    ffi::{c_char, c_double, c_int},
    ptr,
};

use crate::utilies::{FC64, Zero};
//use crate::{g,h};
#[inline]
fn g(l: f64, m: f64, s: f64) -> f64 {
    if l == 0.0 {
        0.0
    } else {
        let a = l.powi(2);
        ((a - m.powi(2)) * (a - s.powi(2)) / (a * (4.0 * a - 1.0))).sqrt()
    }
}
#[inline]
fn h(l: f64, m: f64, s: f64) -> f64 {
    if s == 0.0 || l == 0.0 { 0.0 } else { -m * s / (l * (l + 1.0)) }
}
#[cfg(all(feature = "openblas", not(feature = "mkl")))]
#[link(name = "openblas", kind = "static")]
unsafe extern "C" {
    unsafe fn LAPACKE_dsbevd(matrix_layout: c_int, jobz: c_char, uplo: c_char, n: c_int, KD: c_int, AB: *const c_double, N: c_int, W: *mut c_double, Z: *mut c_double, LDZ: c_int) -> c_int;
    unsafe fn LAPACKE_zgeev(matrix_layout: c_int, jobvl: c_char, jobvr: c_char, n: c_int, a: *const Complex<f64>, lda: c_int, w: *mut Complex<f64>, vl: *mut Complex<f64>, ldvl: c_int, vr: *mut Complex<f64>, ldvr: c_int) -> c_int;
}
#[cfg(feature = "mkl")]
#[link(name = "mkl_intel_lp64", kind = "static")]
#[link(name = "mkl_core", kind = "static")]
#[link(name = "mkl_sequential", kind = "static")]
unsafe extern "C" {
    fn LAPACKE_dsbevd(matrix_layout: c_int, jobz: c_char, uplo: c_char, n: c_int, KD: c_int, AB: *const c_double, N: c_int, W: *mut c_double, Z: *mut c_double, LDZ: c_int) -> c_int;
    fn LAPACKE_zgeev(matrix_layout: c_int, jobvl: c_char, jobvr: c_char, n: c_int, a: *const Complex<f64>, lda: c_int, w: *mut Complex<f64>, vl: *mut Complex<f64>, ldvl: c_int, vr: *mut Complex<f64>, ldvr: c_int) -> c_int;
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
fn sbev2(n: i32, kd: i32, ab: &[f64]) -> (i32, Vec<f64>, Vec<f64>) {
    let mut w: Vec<f64> = Vec::with_capacity(n as usize + 1);
    let mut z: Vec<f64> = Vec::with_capacity((n * n + 3 * n) as usize);
    let result: i32;
    unsafe {
        result = LAPACKE_dsbevd(101, 'V' as c_char, 'U' as c_char, n, kd, ab.as_ptr(), n, w.as_mut_ptr(), z.as_mut_ptr(), n);
        w.set_len(n as usize);
        z.set_len((n * n) as usize);
    }

    (result, w, z)
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
fn geev2(n: i32, a: &[Complex<f64>]) -> (i32, Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let mut w: Vec<Complex<f64>> = Vec::with_capacity(n as usize + 1);
    let mut vr: Vec<Complex<f64>> = Vec::with_capacity((n * n + 3 * n) as usize);
    let result: i32;
    unsafe {
        result = LAPACKE_zgeev(101, 'N' as c_char, 'V' as c_char, n, a.as_ptr(), n, w.as_mut_ptr(), ptr::null_mut::<Complex<f64>>(), 1, vr.as_mut_ptr(), n);
        w.set_len(n as usize);
        vr.set_len((n * n) as usize);
    }

    (result, w, vr)
}

pub fn spec_matrix_upper_ini<T>(l_ini: f64, l: f64, m: f64, s: f64, c: T) -> (Vec<T>, Vec<T>, Vec<T>)
where
    T: FC64,
{
    let mut l_ini = l_ini;
    let ini_size = l + 5.0;
    let mut m2: Vec<T> = vec![Zero::zero(), Zero::zero()];
    let mut m1: Vec<T> = vec![Zero::zero()];
    let mut m0: Vec<T> = Vec::new();

    while l_ini <= ini_size {
        let a = l_ini * (l_ini + 1.0) - s * (s + 1.0);
        let hslm = h(l_ini, m, s);
        let hsl1m = h(l_ini + 1.0, m, s);

        let gslm = g(l_ini, m, s);
        let gsl1m = g(l_ini + 1.0, m, s);
        let gsl2m = g(l_ini + 2.0, m, s);

        let fslm = gsl1m;
        let fsl_1m = gslm;

        m2.push(-c * c * (gsl2m * gsl1m));
        m1.push(-c * c * (gsl1m * (hslm + hsl1m)) + c * (2.0 * s * gsl1m));
        m0.push(-c * c * ((fslm * gsl1m) + gslm * fsl_1m + hslm * hslm) + c * (2.0 * s * hslm) + a);
        l_ini += 1.0;
    }
    (m2, m1, m0)
}
#[inline]
pub fn spec_matrix_extend5<T>(matrix: (&mut Vec<T>, &mut Vec<T>, &mut Vec<T>), l_ini: f64, m: f64, s: f64, c: T)
where
    T: FC64,
{
    let (m2, m1, m0) = matrix;
    let l = m0.len() as f64;
    let mut l_ini = l + l_ini;
    let ini_size = l_ini + 5.0;
    while l_ini <= ini_size {
        let a = l_ini * (l_ini + 1.0) - s * (s + 1.0);
        let hslm = h(l_ini, m, s);
        let hsl1m = h(l_ini + 1.0, m, s);

        let gslm = g(l_ini, m, s);
        let gsl1m = g(l_ini + 1.0, m, s);
        let gsl2m = g(l_ini + 2.0, m, s);

        let fslm = gsl1m;
        let fsl_1m = gslm;

        m2.push(-c * c * (gsl2m * gsl1m));
        m1.push(-c * c * (gsl1m * (hslm + hsl1m)) + c * (2.0 * s * gsl1m));
        m0.push(-c * c * ((fslm * gsl1m) + gslm * fsl_1m + hslm * hslm) + c * (2.0 * s * hslm) + a);
        l_ini += 1.0;
    }
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
pub fn spec_ev_real(matrix: (&Vec<f64>, &Vec<f64>, &Vec<f64>)) -> (i32, Vec<f64>, Vec<f64>) {
    let (m2, m1, m0) = matrix;
    let len = m0.len();
    let mut spec_sb: Vec<f64> = Vec::with_capacity(len * 3);
    spec_sb.extend_from_slice(&m2[0..len]);
    spec_sb.extend_from_slice(&m1[0..len]);
    spec_sb.extend_from_slice(&m0[0..len]);
    sbev2(len as i32, 2, &spec_sb)
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
pub fn spec_ev_complex(matrix: (&Vec<Complex<f64>>, &Vec<Complex<f64>>, &Vec<Complex<f64>>)) -> (i32, Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let (m2, m1, m0) = matrix;
    let len = m0.len();
    let mut spec_sb: Vec<Complex<f64>> = vec![Complex::zero(); len * len];
    spec_sb[0] = m0[0];
    spec_sb[len + 1] = m0[1];
    spec_sb[1] = m1[1];
    spec_sb[len] = m1[1];
    for i in 2..len {
        spec_sb[i + i * len] = m0[i];
        spec_sb[i - 1 + i * len] = m1[i];
        spec_sb[i + (i - 1) * len] = m1[i];
        spec_sb[i - 2 + i * len] = m2[i];
        spec_sb[i + (i - 2) * len] = m2[i];
    }
    geev2(len as i32, &spec_sb)
}
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
pub fn spec_ev_real(matrix: (&Vec<f64>, &Vec<f64>, &Vec<f64>)) -> SelfAdjointEigen<f64> {
    use faer::{Side, mat::Mat};

    let (m2, m1, m0) = matrix;
    let len = m0.len();
    let mut spec_sb = Mat::full(len, len, 0.0);
    spec_sb[(0, 0)] = m0[0];
    spec_sb[(1, 1)] = m0[1];
    spec_sb[(1, 0)] = m1[1];
    spec_sb[(1, 0)] = m1[1];
    for i in 2..len {
        spec_sb[(i, i)] = m0[i];
        spec_sb[(i - 1, i)] = m1[i];
        spec_sb[(i, (i - 1))] = m1[i];
        spec_sb[(i - 2, i)] = m2[i];
        spec_sb[(i, (i - 2))] = m2[i];
    }
    spec_sb.self_adjoint_eigen(Side::Lower).unwrap()
}
#[cfg(all(feature = "faer", not(feature = "mkl"), not(feature = "openblas")))]
pub fn spec_ev_complex(matrix: (&Vec<Complex<f64>>, &Vec<Complex<f64>>, &Vec<Complex<f64>>)) -> Eigen<f64> {
    use crate::const_value::C0;
    use faer::mat::Mat;

    let (m2, m1, m0) = matrix;
    let len = m0.len();
    let mut spec_sb = Mat::full(len, len, C0);
    spec_sb[(0, 0)] = m0[0];
    spec_sb[(1, 1)] = m0[1];
    spec_sb[(1, 0)] = m1[1];
    spec_sb[(1, 0)] = m1[1];
    for i in 2..len {
        spec_sb[(i, i)] = m0[i];
        spec_sb[(i - 1, i)] = m1[i];
        spec_sb[(i, (i - 1))] = m1[i];
        spec_sb[(i - 2, i)] = m2[i];
        spec_sb[(i, (i - 2))] = m2[i];
    }
    spec_sb.eigen().unwrap()
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
#[test]
fn test_real2() {
    let ab = vec![1., 2., 3., 4., 5., 5., 5., 0., 2., 2., 0., 0.];
    let (r, w, z) = sbev2(4, 2, &ab);
    print!("{:?}\n{:?}\n{}\n", w, z, r); //-5.898355 -2.051562 1.096421 10.853496
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
#[test]
fn test_complex2() {
    use crate::const_value::C1;
    let c1 = C1;

    let a = vec![0.0 * c1, -c1, c1, 0.0 * c1];
    let (r, w, z) = geev2(2, &a);
    print!("{:?}\n{:?}\n{}\n", w, z, r); //[Complex { re: 0.0, im: 0.9999999999999997 }, Complex { re: 2.7755575615628914e-17, im: -1.0 }]
}
#[cfg(any(feature = "mkl", feature = "openblas"))]
#[test]
fn test2() {
    let l: f64 = 20.0;
    let m: f64 = 2.0;
    let s: f64 = -2.0;
    let c: f64 = 0.01;

    let (mut m2, mut m1, mut m0) = spec_matrix_upper_ini(m, l, m, s, c);
    spec_matrix_extend5((&mut m2, &mut m1, &mut m0), m, m, s, c);
    let (r, e, v) = spec_ev_real((&m2, &m1, &m0));
    print!("{}\n{:?}\n{:?}\n", r, e, v)
}
