//!local solution for teukolsky radial equation
pub mod poly;
// mod lo_series;
use crate::const_value::{C1, FNLIST, I};
use crate::utilies::*;
use crate::jh::local_solution::poly::PolyCheck;
// use crate::jh::so_series::frac::Frac;
use std::hint::assert_unchecked;

use num_complex::Complex;
use poly::PolyCal;
const MAX_NUM_TERM: usize = 50;
const MAX_NUM_CHECK: usize = 10;

#[inline]
fn zi(p: &Para<f64>, n: f64, a_1: &Complex<f64>, a_2: &Complex<f64>) -> Complex<f64> {
    (I * (p.epsilon * (-2.) * p.kappa * (p.epsilon - p.tau - I * (-1. + n - p.s)) * a_2 + (p.epsilon * p.epsilon - p.lambda + p.epsilon * I * p.kappa * (1. - 2. * n + 2. * p.s) + (-1. + n - p.s - p.tau * I) * (n + p.s - p.tau * I)) * a_1)) / (n * (p.epsilon + p.tau + I * (n - p.s)))
}
#[inline]
fn zo(p: &Para<f64>, n: f64, a_1: &Complex<f64>, a_2: &Complex<f64>) -> Complex<f64> {
    (p.epsilon * (-2.) * p.kappa * (p.epsilon * 2.0 - I * (-1. + n)) * a_2 + (p.epsilon * 2. * p.epsilon * p.kappa - p.lambda + (-1. + n) * (n + 2. * p.s) - p.epsilon * I * (1. - 2. * n - 2. * p.s + p.kappa * (-1. + 2. * n + 2. * (p.tau * I)))) * a_1) / (n * (p.epsilon * I + n + p.s + p.tau * I))
}
#[inline(always)]
pub fn ord(n: usize, cc: &[Complex<f64>; 5], cf: &[f64; 2], a: &[Complex<f64>]) -> Complex<f64> {
    //the most performance sensitive part
    //((cc[0]/n+cc[1]/n/(n-1))*a[0]+(cf[0]+cc[2]/n+cc[3]/n/(n-1))*a[1]+(cf[1]+cc[4]/n)*a[2])
    unsafe { assert_unchecked((3..200).contains(&n)) };
    unsafe { assert_unchecked(a.len() == 3) };
    let n_recip = FNLIST[n]; // 1/n
    let nsub1_recip = FNLIST[n - 1]; // 1/(n-1)
    let nf = n as f64; // n
    // * (mul) usually is faster than / (div)
    //by extracting a factor 1/n, we change a * for (Complex<f64>,f64) to a * for (f64,f64)
    #[cfg(not(all(target_feature = "sse2", target_feature = "sse3")))]
    {
        //by extracting a factor 1/n, we change a * for (Complex<f64>,f64) to a * for (f64,f64)
        ((cc[0] + cc[2] * nsub1_recip) * a[0]
            + (cf[0] * nf  + cc[1]//here :)
            + cc[3] * nsub1_recip)
                * a[1]
            + (cf[1] * nf + cc[4]) * a[2])
            * n_recip
    }

    #[cfg(all(target_feature = "sse2", target_feature = "sse3"))]
    unsafe {
        use std::arch::x86_64::{_mm_loadu_pd, _mm_set1_pd};
        #[cfg(target_feature = "fma")]
        {
            use std::arch::x86_64::{_mm_addsub_pd, _mm_fmadd_pd, _mm_mul_pd, _mm_shuffle_pd};
            use std::mem::transmute;
            use std::ptr::addr_of;
            let cc0_pd = _mm_loadu_pd(cc.as_ptr().offset(0) as *const f64);
            let cc1_p_cf0_t_nf = cc[1] + cf[0] * nf;
            let cc1_p_cf0_t_nf_pd = _mm_loadu_pd(addr_of!(cc1_p_cf0_t_nf) as *const f64);
            let cc2_pd = _mm_loadu_pd(cc.as_ptr().offset(2) as *const f64);
            let cc3_pd = _mm_loadu_pd(cc.as_ptr().offset(3) as *const f64);
            let cc4_p_cf1_t_nf = cc[4] + cf[1] * nf;
            let coe_a2 = _mm_loadu_pd(addr_of!(cc4_p_cf1_t_nf) as *const f64);

            let nsub1_recip = _mm_set1_pd(nsub1_recip);

            let coe_a0 = _mm_fmadd_pd(cc2_pd, nsub1_recip, cc0_pd);
            let coe_a1 = _mm_fmadd_pd(cc3_pd, nsub1_recip, cc1_p_cf0_t_nf_pd);

            let rtpart = _mm_fmadd_pd(_mm_set1_pd(a[2].re), coe_a2, _mm_fmadd_pd(_mm_set1_pd(a[0].re), coe_a0, _mm_mul_pd(_mm_set1_pd(a[1].re), coe_a1)));
            let itpart = _mm_fmadd_pd(_mm_set1_pd(a[2].im), coe_a2, _mm_fmadd_pd(_mm_set1_pd(a[0].im), coe_a0, _mm_mul_pd(_mm_set1_pd(a[1].im), coe_a1)));

            transmute::<_, Complex<f64>>(_mm_mul_pd(_mm_set1_pd(n_recip), _mm_addsub_pd(rtpart, _mm_shuffle_pd::<1>(itpart, itpart))))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            // (coa1 * a[0]+ coa2part*a[1]+cc[4]*a[2]) * n_recip + (cf[0]*a[1]+cf[1]*a[2])
            use std::arch::x86_64::{_mm_add_pd, _mm_addsub_pd, _mm_mul_pd, _mm_shuffle_pd};
            use std::mem::transmute;
            use std::ptr::addr_of;
            let cc0_pd = _mm_loadu_pd(cc.as_ptr().offset(0) as *const f64);
            let cc1_p_cf0_t_nf = cc[1] + cf[0] * nf;
            let cc1_p_cf0_t_nf_pd = _mm_loadu_pd(addr_of!(cc1_p_cf0_t_nf) as *const f64);
            let cc2_pd = _mm_loadu_pd(cc.as_ptr().offset(2) as *const f64);
            let cc3_pd = _mm_loadu_pd(cc.as_ptr().offset(3) as *const f64);
            let cc4_p_cf1_t_nf = cc[4] + cf[1] * nf;
            let coe_a2 = _mm_loadu_pd(addr_of!(cc4_p_cf1_t_nf) as *const f64);

            let nsub1_recip = _mm_set1_pd(nsub1_recip);

            let coe_a0 = _mm_add_pd(_mm_mul_pd(cc2_pd, nsub1_recip), cc0_pd);
            let coe_a1 = _mm_add_pd(_mm_mul_pd(cc3_pd, nsub1_recip), cc1_p_cf0_t_nf_pd);

            let rtpart = _mm_add_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(a[0].re), coe_a0), _mm_mul_pd(_mm_set1_pd(a[1].re), coe_a1)), _mm_mul_pd(_mm_set1_pd(a[2].re), coe_a2));
            let itpart = _mm_add_pd(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(a[0].im), coe_a0), _mm_mul_pd(_mm_set1_pd(a[1].im), coe_a1)), _mm_mul_pd(_mm_set1_pd(a[2].im), coe_a2));

            transmute::<_, Complex<f64>>(_mm_mul_pd(_mm_set1_pd(n_recip), _mm_addsub_pd(rtpart, _mm_shuffle_pd::<1>(itpart, itpart))))
        }
    }
}
#[inline]
fn ii(p: &Para<f64>, n: f64, a_1: &Complex<f64>, a_2: &Complex<f64>) -> Complex<f64> {
    (p.epsilon * 2. * p.kappa * n).recip() * (I * (-((p.epsilon * 2. - I * (-1. + n)) * (p.epsilon - p.tau - I * (-1. + n - p.s)) * a_2) + (p.epsilon * 2. * p.epsilon * p.kappa + p.lambda + n - n * n + 2. * n * p.s - p.epsilon * I * (-1. + 2. * n - 2. * p.s + p.kappa * (-1. + 2. * n - 2. * (p.tau * I)))) * a_1))
}
#[inline]
fn io(p: &Para<f64>, n: f64, a_1: &Complex<f64>, a_2: &Complex<f64>) -> Complex<f64> {
    -((p.epsilon * 2. * p.kappa * n).recip() * ((-1. + n + 2. * p.s) * (p.epsilon + p.tau + I * (-1. + n + p.s)) * a_2 + ((-p.epsilon) * (1. + p.kappa) * (-1. + 2. * n + 2. * p.s) + (p.lambda - (-1. + n) * (n + 2. * p.s)) * I) * a_1))
}

///the set of parameters for teukolsky radial equation and local solutions.
pub struct Para<T> {
    pub s: f64,
    pub m: f64,
    pub a: f64,
    pub omega: T,
    pub lambda: T,

    pub tau: T,
    pub epsilon: T,
    pub kappa: f64,
    // nu: Option<Complex<f64>>,
}
impl Para<f64> {
    ///generate the set of parameters for teukolsky radial equation and local solutions.
    pub fn new(s: f64, m: f64, a: f64, omega: f64, lambda: f64) -> Self {
        let epsilon = omega * 2.;
        let kappa = (1. - a * a).sqrt();
        let tau = (epsilon - m * a) / kappa;

        // let nu;
        // if omega.abs() < 0.01 {
        //     nu = Some(C0);
        // } else {
        //     nu = None;
        // }
        Para { s, m, a, omega, tau, lambda, epsilon, kappa }
    }
    ///cache for ordinary point local solution, which is the performance bottleneck in our series expansion method.
    pub fn ord_cache(&self) -> [Complex<f64>; 6] {
        let mut cache: [Complex<f64>; 6] = Default::default();
        cache[0] = -2.0 * (self.epsilon * I) * self.kappa;
        cache[1] = self.epsilon * self.kappa * 2.0 * (self.epsilon - self.tau + I * (1.0 + self.s));

        cache[2] = -2. - 2. * (self.epsilon * I) * self.kappa - 2. * (self.tau * I);
        cache[3] = self.epsilon * self.epsilon - self.lambda + self.epsilon * self.kappa * (1.0 + 2.0 * self.s) * I - (self.s - self.tau * I) * (self.tau * I + 1.0 + self.s);
        cache[4] = -2. - 2. * (self.tau * I);
        cache[5] = 1. + self.s + (self.epsilon + self.tau) * I;
        cache
    }
}

impl Para<f64> {
    #[inline]
    fn residual_zi(&self, sol: ResN<3, Complex<f64>>, x: f64) -> f64 {
        let ResN([p0, p1, p2]) = sol;
        let c0 = ((-self.epsilon.powi(2) + self.lambda + self.s + self.s.powi(2) + self.tau.powi(2) + self.tau * I + self.epsilon * self.kappa * I * (1. - 2. * self.s)) + (self.epsilon * self.kappa * 2. * (self.epsilon - self.tau + I * (self.s - 1.))) * x) * p0;
        let c1 = ((1. - self.s - (self.epsilon + self.tau) * I) + (2. * (self.epsilon * self.kappa * I + self.tau * I - 1.) - (self.epsilon * self.kappa * 2.0 * I) * x) * x) * p1;
        let c2 = x * (1. - x) * p2;

        (c0 + c1 + c2).abs() / c0.abs().max(c1.abs().max(c2.abs()))
    }
    #[inline]
    fn residual_zo(&self, sol: ResN<3, Complex<f64>>, x: f64) -> f64 {
        let ResN([p0, p1, p2]) = sol;
        let c0 = (self.lambda + self.epsilon.powi(2) * 2. * self.kappa * (-1. + 2. * x) - self.epsilon * I * (1. + 2. * self.s + self.kappa * (-1. - self.tau * 2. * I + 2. * x))) * p0;
        let c1 = (1. + self.s + self.tau * I - 2. * (1. + self.s) * x - self.epsilon * I * (-1. + 2. * (1. + self.kappa * (-1. + x)) * x)) * p1;
        let c2 = x * (1. - x) * p2;

        (c0 + c1 + c2).abs() / c0.abs().max(c1.abs().max(c2.abs()))
    }
    #[inline]
    fn residual_ii(&self, sol: ResN<3, Complex<f64>>, x: f64) -> f64 {
        let ResN([p0, p1, p2]) = sol;

        let c0 = (self.epsilon.powi(2) * 2. * self.kappa + self.lambda - self.epsilon * self.tau * 2. * self.kappa - self.epsilon * I * (self.kappa + 1. - 2. * self.s) + 2. * self.s - (self.epsilon * 2. - I) * (self.epsilon + I * (-1. + self.s + self.tau * I)) / x) * p0;
        let c1 = (-self.epsilon * 3. * I - 1. + self.s + self.tau * I + (self.epsilon * 2. * I * (1. + self.kappa) - 2. * self.s) * x - self.epsilon * 2. * I * self.kappa * x.powi(2)) * p1;
        let c2 = x * (1. - x) * p2;

        (c0 + c1 + c2).abs() / c0.abs().max(c1.abs().max(c2.abs()))
    }
    #[inline]
    fn residual_io(&self, sol: ResN<3, Complex<f64>>, x: f64) -> f64 {
        let ResN([p0, p1, p2]) = sol;

        let c0 = (self.lambda + self.epsilon * I * (1. + self.kappa) * (1. + 2. * self.s) + (1. + 2. * self.s) * (1. - self.epsilon * I + self.s - self.tau * I) / x) * p0;
        let c1 = (I * (self.epsilon + self.tau + I + 3. * I * self.s) + (-self.epsilon * 2. * I * (1. + self.kappa) + 2. * self.s) * x + self.epsilon * 2. * I * self.kappa * x.powi(2)) * p1;
        let c2 = x * (1. - x) * p2;

        (c0 + c1 + c2).abs() / c0.abs().max(c1.abs().max(c2.abs()))
    }
}

///initialize the local solution
pub trait Init
where
    Self: Sized,
{
    //this function cannot be used to generate a ordinary point local solution
    fn init_gen(p: &Para<f64>, num: usize) -> Self;
}
///calculate the local solution
pub trait Cal<P>
where
    Self: Sized + Range,
{
    ///calculate the local solution at x
    fn cal(&self, p: &P, x: f64) -> NormResC64;
    fn cal3(&self, p: &P, x: f64) -> ResN<3, Complex<f64>>;
}
///give the convergence range of local solution
pub trait Range {
    ///give left position of the convergence range
    fn left(&self) -> f64;
    ///give right position of the convergence range
    fn right(&self) -> f64;
}
///local solution at 0, which correspond to the ingoing bound condition
#[derive(Default)]
pub struct ZeroIn {
    coe: Vec<Complex<f64>>,
    pub range: f64,
}

///local solution at 0, which correspond to the outgoing bound condition
#[derive(Default)]
pub struct ZeroOut {
    coe: Vec<Complex<f64>>,
    range: f64,
}
///local solution at ordinary point x0
#[derive(Default)]
pub struct Ordinary {
    pub x0: f64,
    exp2: i64,
    vec_ie: (usize, usize),
    range: f64,
}
///local solution at infinity, which correspond to the ingoing bound condition
#[derive(Default)]
pub struct InfinityIn {
    coe: Vec<Complex<f64>>,
    pub range: f64,
}
///local solution at infinity, which correspond to the outgoing bound condition
#[derive(Default)]
pub struct InfinityOut {
    coe: Vec<Complex<f64>>,
    range: f64,
}
impl HeapSize for ZeroIn {
    fn get_heapsize(&self) -> usize {
        self.coe.get_heapsize()
    }
}
impl HeapSize for ZeroOut {
    fn get_heapsize(&self) -> usize {
        self.coe.get_heapsize()
    }
}
impl HeapSize for InfinityIn {
    fn get_heapsize(&self) -> usize {
        self.coe.get_heapsize()
    }
}
impl HeapSize for InfinityOut {
    fn get_heapsize(&self) -> usize {
        self.coe.get_heapsize()
    }
}
impl Init for ZeroIn {
    ///initialize the local solution at 0 with the ingoing bound condition
    #[inline]
    fn init_gen(p: &Para<f64>, num: usize) -> Self {
        let c1 = (p.epsilon * (p.kappa - 2. * p.kappa * p.s) + p.epsilon * p.epsilon * I - I * (p.lambda + p.s + p.s * p.s + p.tau * (p.tau + I))) / (p.tau + p.epsilon + I - I * p.s);
        let mut coe = Vec::with_capacity(num + 5);
        coe.push(C1);
        coe.push(c1);
        ZeroIn { coe, range: 0.0 }
    }
}
impl Init for ZeroOut {
    ///initialize the local solution at 0 with the outgoing bound condition
    #[inline]
    fn init_gen(p: &Para<f64>, num: usize) -> Self {
        let c1 = (p.epsilon * p.epsilon * 2. * p.kappa - p.lambda - p.epsilon * I * (-1. - 2. * p.s + p.kappa * (1. + 2. * (p.tau * I)))) / (1. + p.epsilon * I + p.s + p.tau * I);
        let mut coe = Vec::with_capacity(num + 5);
        coe.push(C1);
        coe.push(c1);
        ZeroOut { coe, range: 0.0 }
    }
}
impl Ordinary {
    ///initialize the local solution  at ordinary point x0
    #[inline]
    fn init_gen_x0(vec: &mut Vec<Complex<f64>>, cache: &[Complex<f64>; 6], x0: f64, v: NormResC64, _num: usize) -> (Self, [Complex<f64>; 5], [f64; 2]) {
        //most performance sensitive part, I should not trust the compiler here :(
        let init = vec.len();

        let xx1 = x0 * (x0 - 1.0);
        let x1 = x0 - 1.0;
        let ixx1 = xx1.recip();
        // let ix = (x0-1.0).recip();
        let mut c: [Complex<f64>; 5] = Default::default();
        let cf = [-ixx1, ixx1 - 2.0 / x1];

        c[0] = cache[0] / xx1;
        c[2] = cache[1] / xx1;
        c[1] = 2.0 * cache[0] / x1 - cache[2] / xx1;
        c[3] = cache[1] / x1 - cache[3] / xx1;
        c[4] = cache[0] - cache[4] / x1 - cache[5] / xx1;

        let c2 = v.res.1 * (cf[1] + c[4] / 2.0) + (cf[0] + c[1] / 2.0 + c[3] / 2.0) * v.res.0;

        vec.push(v.res.0);
        vec.push(v.res.1);
        vec.push(c2);

        (Ordinary { x0, exp2: v.exp2, vec_ie: (init, 0), range: 0.0 }, c, cf)
    }
    // #[inline]
    // fn len(&self)->usize{
    //     self.vec_ie.1-self.vec_ie.0
    // }
}
impl Init for InfinityIn {
    ///initialize the local solution at infinity with the ingoing bound condition
    #[inline]
    fn init_gen(p: &Para<f64>, num: usize) -> Self {
        let c1 = (p.epsilon * 2. * p.kappa).recip() * ((p.lambda + 2. * p.s) * I + p.epsilon * (1. + p.kappa + p.epsilon * 2. * I * p.kappa - 2. * p.s - p.tau * p.kappa * 2. * I));
        let mut coe = Vec::with_capacity(num + 5);
        coe.push(C1);
        coe.push(c1);
        InfinityIn { coe, range: f64::INFINITY }
    }
}
impl Init for InfinityOut {
    ///initialize the local solution at infinity with the ingoing bound condition
    #[inline]
    fn init_gen(p: &Para<f64>, num: usize) -> Self {
        let c1 = (p.epsilon * 2. * p.kappa).recip() * (p.epsilon * (1. + p.kappa) * (1. + 2. * p.s) - p.lambda * I);
        let mut coe = Vec::with_capacity(num + 5);
        coe.push(C1);
        coe.push(c1);
        InfinityOut { coe, range: f64::INFINITY }
    }
}
impl Cal<Para<f64>> for ZeroIn {
    #[inline]
    fn cal(&self, _p: &Para<f64>, x: f64) -> NormResC64 {
        debug_assert!(x <= 0.0 && x >= self.left(), "{x}\n{}", self.left());
        NormResC64 { exp2: 0, res: self.coe.res(x) }
    }
    fn cal3(&self, _p: &Para<f64>, x: f64) -> ResN<3, Complex<f64>> {
        self.coe.res3(x)
    }
}
impl Cal<Para<f64>> for ZeroOut {
    #[inline]
    fn cal(&self, p: &Para<f64>, x: f64) -> NormResC64 {
        debug_assert!(x <= 0.0 && x >= self.left());

        let v = self.coe.res(x);
        let mut exp = NormResC64::from_expfx((-x).ln() * (I * (p.epsilon + p.tau - I * p.s)), (I * (p.epsilon + p.tau - I * p.s)) / x);
        exp *= v;
        exp
    }
    fn cal3(&self, p: &Para<f64>, x: f64) -> ResN<3, Complex<f64>> {
        let v = self.coe.res3(x);
        let mut exp = ResN::<3, Complex<f64>>::exp_fx((-x).ln() * (I * (p.epsilon + p.tau - I * p.s)), (I * (p.epsilon + p.tau - I * p.s)) / x, -(I * (p.epsilon + p.tau - I * p.s)) / x / x);
        exp *= &v;
        exp
    }
}
impl Cal<Vec<Complex<f64>>> for Ordinary {
    #[inline(always)]
    fn cal(&self, vec: &Vec<Complex<f64>>, x: f64) -> NormResC64 {
        debug_assert!(x <= self.right() && x >= self.left());

        let x = x - self.x0;

        //cancel the bound check, i'am very sure that the following conditions are always be satisfied :)
        unsafe { assert_unchecked(self.vec_ie.1 <= vec.len()) };
        unsafe { assert_unchecked(self.vec_ie.0 < self.vec_ie.1) };

        let res = vec[self.vec_ie.0..self.vec_ie.1].res(x);
        NormResC64 { exp2: self.exp2, res }
    }
    fn cal3(&self, vec: &Vec<Complex<f64>>, x: f64) -> ResN<3, Complex<f64>> {
        let x = x - self.x0;

        let mut res = vec[self.vec_ie.0..self.vec_ie.1].res3(x);
        res[0] *= 2.0_f64.powi(self.exp2 as i32);
        res[1] *= 2.0_f64.powi(self.exp2 as i32);
        res[2] *= 2.0_f64.powi(self.exp2 as i32);
        res
    }
}
impl Cal<Para<f64>> for InfinityIn {
    #[inline]
    fn cal(&self, p: &Para<f64>, x: f64) -> NormResC64 {
        debug_assert!(x <= self.right() && x <= 0.0);

        let mut v = self.coe.res(1.0 / x);
        v.1 /= -x.powi(2);
        let mut exp = NormResC64::from_expfx((-x).ln() * (-1. - p.epsilon * I + p.s + p.tau * I), (-1. - p.epsilon * I + p.s + p.tau * I) / x);
        exp *= v;
        exp
    }
    fn cal3(&self, p: &Para<f64>, x: f64) -> ResN<3, Complex<f64>> {
        let mut v = self.coe.res3(1.0 / x);
        v[1] /= -x.powi(2);
        v[2] /= x.powi(4);
        let part = v[1] / x * (-2.0);
        v[2] += part;
        let mut exp = ResN::exp_fx((-x).ln() * (-1. - p.epsilon * I + p.s + p.tau * I), (-1. - p.epsilon * I + p.s + p.tau * I) / x, -(-1. - p.epsilon * I + p.s + p.tau * I) / x / x);
        exp *= &v;
        exp
    }
}
impl Cal<Para<f64>> for InfinityOut {
    #[inline]
    fn cal(&self, p: &Para<f64>, x: f64) -> NormResC64 {
        debug_assert!(x <= self.right() && x <= 0.0);

        let mut v = self.coe.res(1.0 / x);
        v.1 /= -x.powi(2);
        let mut exp = NormResC64::from_expfx((-x).ln() * (-1. + p.epsilon * I - p.s + p.tau * I) + p.epsilon * (-2.) * I * p.kappa * x, p.epsilon * (-2.) * I * p.kappa + (-1. + p.epsilon * I - p.s + p.tau * I) / x);
        exp *= v;
        exp
    }
    fn cal3(&self, p: &Para<f64>, x: f64) -> ResN<3, Complex<f64>> {
        let mut v = self.coe.res3(1.0 / x);
        v[1] /= -x.powi(2);
        v[2] /= x.powi(4);
        let part = v[1] / x * (-2.0);
        v[2] += part;
        let mut exp = ResN::exp_fx((-x).ln() * (-1. + p.epsilon * I - p.s + p.tau * I) + p.epsilon * (-2.) * I * p.kappa * x, p.epsilon * (-2.) * I * p.kappa + (-1. + p.epsilon * I - p.s + p.tau * I) / x, -(-1. + p.epsilon * I - p.s + p.tau * I) / x / x);
        exp *= &v;
        exp
    }
}
impl ZeroIn {
    #[inline]
    fn check(&mut self, p: &Para<f64>, err: f64) -> bool {
        let x = -self.range;
        let sol = self.coe.res3(x);
        let err_estimate_new = p.residual_zi(sol, x);
        if err_estimate_new.abs() < 10.0 * err {
            self.coe.check_res(x, err);
            true
        } else {
            self.range *= 0.9;
            false
        }
    }
    pub fn new(p: &Para<f64>, num_min: usize, err: f64) -> Self {
        let mut zeroi = ZeroIn::init_gen(p, num_min);
        loop {
            let n = zeroi.coe.len();
            let value = unsafe { zi(p, n as f64, zeroi.coe.get_unchecked(n - 1), zeroi.coe.get_unchecked(n - 2)) };
            if value.is_infinite() {
                if zeroi.range == 0.0 {
                    zeroi.range = fast_rootn_approx(err / zeroi.coe.last().unwrap().abs(), (n - 1) as i32);
                }
                break;
            }
            let range_new = fast_rootn_approx(err / value.abs(), (n - 1) as i32);
            if (n > num_min && range_new > 0.6_f64) || n > MAX_NUM_TERM {
                zeroi.range = range_new;
                break;
            }
            zeroi.coe.push(value);
        }
        for _ in 0..MAX_NUM_CHECK {
            if zeroi.check(p, err) {
                break;
            }
        }
        zeroi.range = zeroi.range.min(0.6);
        zeroi
    }
}
impl ZeroOut {
    #[inline]
    fn check(&mut self, p: &Para<f64>, err: f64) -> bool {
        let x = -self.range;
        let sol = self.coe.res3(x);
        let err_estimate_new = p.residual_zo(sol, x);
        if err_estimate_new.abs() < 10.0 * err {
            self.coe.check_res(x, err);
            true
        } else {
            self.range *= 0.9;
            false
        }
    }
    pub fn new(p: &Para<f64>, num_min: usize, err: f64) -> Self {
        let mut zeroo = ZeroOut::init_gen(p, num_min);
        loop {
            let n = zeroo.coe.len();
            let value = unsafe { zo(p, n as f64, zeroo.coe.get_unchecked(n - 1), zeroo.coe.get_unchecked(n - 2)) };
            if value.is_infinite() {
                if zeroo.range == 0.0 {
                    zeroo.range = fast_rootn_approx(err / zeroo.coe.last().unwrap().abs(), (n - 1) as i32);
                }
                break;
            }
            let range_new = fast_rootn_approx(err / value.abs(), (n - 1) as i32);
            if (n > num_min && range_new > 0.6_f64) || n > MAX_NUM_TERM {
                zeroo.range = range_new;
                break;
            }
            zeroo.coe.push(value);
        }
        for _ in 0..MAX_NUM_CHECK {
            if zeroo.check(p, err) {
                break;
            }
        }
        zeroo.range = zeroo.range.min(0.6);
        zeroo
    }
}
impl Ordinary {
    #[inline]
    fn check(&mut self, p: &Para<f64>, vec: &mut Vec<Complex<f64>>, err: f64) -> bool {
        let x = -self.range;
        let sol = vec[self.vec_ie.0..self.vec_ie.1].res3(x);
        let err_estimate_new = p.residual_zi(sol, self.x0 + x);
        if err_estimate_new.abs() < 10.0 * err {
            // vec.check_res_with_init(x, self.vec_ie.0, err);
            self.vec_ie.1 = vec.len();
            true
        } else {
            self.range *= 0.8;
            false
        }
    }
    ///generate a power series solution at ```x0``` with degree ```num``` and initial condition ```v```
    pub fn new_x0(p: &Para<f64>, x0: f64, vec: &mut Vec<Complex<f64>>, cache: &[Complex<f64>; 6], res: NormResC64, num: usize, err: f64) -> Self {
        let (mut sol, cc, cf) = Self::init_gen_x0(vec, cache, x0, res, num);
        let num = num.min(200);
        for i in 3..num {
            let value = ord(i, &cc, &cf, &vec[vec.len() - 3..vec.len()]);
            if value.re.abs() > 2.0_f64.powi(500) || value.im.abs() > 2.0_f64.powi(500) || value.re.abs() < 2.0_f64.powi(-500) || value.im.abs() < 2.0_f64.powi(-500) {
                break;
            }
            vec.push(value);
        }
        sol.vec_ie.1 = vec.len();

        sol.range = fast_rootn_approx(err / unsafe { vec.last().unwrap_unchecked().abs() }, (sol.vec_ie.1 - sol.vec_ie.0) as i32);
        for _ in 0..MAX_NUM_CHECK {
            if sol.check(p, vec, err) {
                break;
            }
        }
        sol
    }
}
impl InfinityIn {
    #[inline]
    fn check(&mut self, p: &Para<f64>, err: f64) -> bool {
        let x = -self.range;
        let mut sol = self.coe.res3(x.recip());
        sol[1] /= -x.powi(2);
        sol[2] /= x.powi(4);
        let part = sol[1] / x * (-2.0);
        sol[2] += part;
        let err_estimate_new = p.residual_ii(sol, x);
        if err_estimate_new.abs() < 10.0 * err {
            self.coe.check_res(x.recip(), err);
            true
        } else {
            self.range *= 1.1;
            false
        }
    }
    pub fn new(p: &Para<f64>, num_min: usize, err: f64) -> Self {
        let mut infi = InfinityIn::init_gen(p, num_min);
        loop {
            let n = infi.coe.len();
            let value = unsafe { ii(p, n as f64, infi.coe.get_unchecked(n - 1), infi.coe.get_unchecked(n - 2)) };
            if value.is_infinite() {
                if infi.range.is_infinite() {
                    infi.range = fast_rootn_approx(err / infi.coe.last().unwrap().abs(), (n - 1) as i32).recip();
                }
                break;
            }
            let range_new = fast_rootn_approx(err / value.abs(), (n - 1) as i32).recip();
            if n > num_min {
                if infi.range > range_new {
                    infi.range = range_new;
                } else {
                    break;
                }
            }
            if n > MAX_NUM_TERM {
                infi.range = range_new;
                break;
            }
            infi.coe.push(value);
        }
        for _ in 0..MAX_NUM_CHECK {
            if infi.check(p, err) {
                break;
            }
        }
        infi
    }
}
impl InfinityOut {
    #[inline]
    fn check(&mut self, p: &Para<f64>, err: f64) -> bool {
        let x = -self.range;
        let mut sol = self.coe.res3(x.recip());
        sol[1] /= -x.powi(2);
        sol[2] /= x.powi(4);
        let part = sol[1] / x * (-2.0);
        sol[2] += part;
        let err_estimate_new = p.residual_io(sol, x);
        if err_estimate_new.abs() < 10.0 * err {
            self.coe.check_res(x.recip(), err);
            true
        } else {
            self.range *= 1.1;
            false
        }
    }
    pub fn new(p: &Para<f64>, num_min: usize, err: f64) -> Self {
        let mut info = InfinityOut::init_gen(p, num_min);

        loop {
            let n = info.coe.len();
            let value = unsafe { io(p, n as f64, info.coe.get_unchecked(n - 1), info.coe.get_unchecked(n - 2)) };
            if value.is_infinite() {
                if info.range.is_infinite() {
                    info.range = fast_rootn_approx(err / info.coe.last().unwrap().abs(), (n - 1) as i32).recip();
                }
                break;
            }
            let range_new = fast_rootn_approx(err / value.abs(), (n - 1) as i32).recip();
            if n > num_min {
                if info.range > range_new {
                    info.range = range_new;
                } else {
                    break;
                }
            }
            if n > MAX_NUM_TERM {
                info.range = range_new;
                break;
            }
            info.coe.push(value);
        }
        for _ in 0..MAX_NUM_CHECK {
            if info.check(p, err) {
                break;
            }
        }

        info
    }
}
impl Range for ZeroIn {
    #[inline]
    fn left(&self) -> f64 {
        -self.range
    }
    #[inline]
    fn right(&self) -> f64 {
        self.range
    }
}
impl Range for ZeroOut {
    #[inline]
    fn left(&self) -> f64 {
        -self.range
    }
    #[inline]
    fn right(&self) -> f64 {
        self.range
    }
}
impl Range for Ordinary {
    #[inline]
    fn left(&self) -> f64 {
        self.x0 - self.range
    }
    #[inline]
    fn right(&self) -> f64 {
        self.x0 + self.range
    }
}
impl Range for InfinityIn {
    #[inline]
    fn left(&self) -> f64 {
        f64::NEG_INFINITY
    }
    #[inline]
    fn right(&self) -> f64 {
        -self.range
    }
}
impl Range for InfinityOut {
    #[inline]
    fn left(&self) -> f64 {
        f64::NEG_INFINITY
    }
    #[inline]
    fn right(&self) -> f64 {
        -self.range
    }
}
