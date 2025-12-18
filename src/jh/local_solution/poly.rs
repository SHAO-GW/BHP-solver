//! calculate the polynomial
// Burrus, C.S., Fox, J.W., Sitton, G.A., & Treitel, S. (2003). Horner's Method for Evaluating and Deflating Polynomials.

use crate::const_value::C0;
use crate::utilies::{Res, ResN};
use num_complex::Complex;
use std::arch::x86_64::{_mm_add_pd, _mm_fmadd_pd, _mm_mul_pd, _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_loadu2_m128d, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd};
use std::hint::{assert_unchecked, select_unpredictable};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
use std::{
    arch::x86_64::{_mm_loadu_pd, _mm_set1_pd, _mm_storeu_pd},
    ptr::{addr_of, addr_of_mut},
};
//__________________________________________________________________________
pub trait PolyCal<T> {
    type Out;
    #[cfg(test)]
    fn v(self, x: T) -> Self::Out;
    #[cfg(test)]
    fn dvdx(self, x: T) -> Self::Out;

    fn res(self, x: T) -> Res<Self::Out>;
    fn res3(self, x: T) -> ResN<3, Self::Out>;
}
impl PolyCal<f64> for &[Complex<f64>] {
    type Out = Complex<f64>;
    #[cfg(test)]
    fn v(self, x: f64) -> Self::Out {
        let mut ans: Self::Out = C0;
        for &i in self.iter().skip(1).rev() {
            ans += i;
            ans *= x;
        }
        ans += unsafe { self.get_unchecked(0) };
        ans
    }
    #[cfg(test)]
    fn dvdx(self, x: f64) -> Self::Out {
        let mut ans: Self::Out = C0;
        for (&i, n) in (self.iter().skip(2).zip(2..self.len())).rev() {
            ans += i * n as f64;
            ans *= x;
        }
        ans += unsafe { self.get_unchecked(1) };
        ans
    }
    #[inline]
    fn res(self, x: f64) -> Res<Self::Out> {
        unsafe { assert_unchecked(!self.is_empty()) };

        unsafe {
            if cfg!(target_feature = "avx") {
                if cfg!(target_feature = "fma") { poly2_avx_fma(self, x) } else { poly2_avx(self, x) }
            } else if cfg!(target_feature = "sse2") {
                if cfg!(target_feature = "fma") { poly2_sse2_fma(self, x) } else { poly2_sse2(self, x) }
            } else {
                poly2(self, x)
            }
        }
    }
    #[inline]
    fn res3(self, x: f64) -> ResN<3, Self::Out> {
        unsafe { assert_unchecked(!self.is_empty()) };

        unsafe {
            if cfg!(target_feature = "avx") {
                if cfg!(target_feature = "fma") { poly3_avx_fma(self, x) } else { poly3_avx(self, x) }
            } else if cfg!(target_feature = "sse2") {
                if cfg!(target_feature = "fma") { poly3_sse2_fma(self, x) } else { poly3_sse2(self, x) }
            } else {
                poly3(self, x)
            }
        }
    }
}
#[inline]
fn poly2(coe: &[Complex<f64>], x: f64) -> Res<Complex<f64>> {
    let (mut f, mut fp) = (coe[coe.len() - 1], C0);

    for &i in coe.iter().rev().skip(1) {
        fp.re = fp.re.mul_add(x, f.re);
        fp.im = fp.im.mul_add(x, f.im);

        f.re = f.re.mul_add(x, i.re);
        f.im = f.im.mul_add(x, i.im);
    }
    Res(f, fp)
}
#[inline]
#[target_feature(enable = "sse2")]
fn poly2_sse2(coe: &[Complex<f64>], x: f64) -> Res<Complex<f64>> {
    let (mut f, mut fp) = (coe[coe.len() - 1], C0);
    let (x_pd, mut f_pd, mut fp_pd) = unsafe { (_mm_set1_pd(x), _mm_loadu_pd(addr_of!(f) as *const f64), _mm_set1_pd(0.0)) };

    for &i in coe.iter().rev().skip(1) {
        let i_pd = unsafe { _mm_loadu_pd(addr_of!(i) as *const f64) };
        fp_pd = _mm_add_pd(_mm_mul_pd(fp_pd, x_pd), f_pd);
        f_pd = _mm_add_pd(_mm_mul_pd(f_pd, x_pd), i_pd);
    }
    unsafe {
        _mm_storeu_pd(addr_of_mut!(f) as *mut f64, f_pd);
        _mm_storeu_pd(addr_of_mut!(fp) as *mut f64, fp_pd);
    }
    Res(f, fp)
}
#[inline]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "fma")]
fn poly2_sse2_fma(coe: &[Complex<f64>], x: f64) -> Res<Complex<f64>> {
    let (mut f, mut fp) = (coe[coe.len() - 1], C0);
    let (x_pd, mut f_pd, mut fp_pd) = unsafe { (_mm_set1_pd(x), _mm_loadu_pd(addr_of!(f) as *const f64), _mm_set1_pd(0.0)) };

    for &i in coe.iter().rev().skip(1) {
        let i_pd = unsafe { _mm_loadu_pd(addr_of!(i) as *const f64) };
        fp_pd = _mm_fmadd_pd(fp_pd, x_pd, f_pd);
        f_pd = _mm_fmadd_pd(f_pd, x_pd, i_pd);
    }
    unsafe {
        _mm_storeu_pd(addr_of_mut!(f) as *mut f64, f_pd);
        _mm_storeu_pd(addr_of_mut!(fp) as *mut f64, fp_pd);
    }
    Res(f, fp)
}
#[inline]
#[target_feature(enable = "avx")]
fn poly2_avx(coe: &[Complex<f64>], x: f64) -> Res<Complex<f64>> {
    let coe_ptr = coe.as_ptr() as *const f64;

    let x_pd = _mm256_set1_pd(x * x);
    let mut f_pd;
    let offset;
    if coe.len() % 2 == 0 {
        f_pd = unsafe { _mm256_loadu_pd(addr_of!(coe[coe.len() - 2]) as *const f64) };
        offset = 2 * (coe.len() as isize - 2);
    } else {
        let c0 = C0;
        f_pd = unsafe { _mm256_loadu2_m128d(addr_of!(c0) as *const f64, addr_of!(coe[coe.len() - 1]) as *const f64) };
        offset = 2 * (coe.len() as isize - 1);
    }
    let mut fp_pd = _mm256_set1_pd(0.0);
    for i in 1..coe.len().div_ceil(2) {
        let i_pd = unsafe { _mm256_loadu_pd(coe_ptr.offset(offset - i as isize * 4)) };
        fp_pd = _mm256_add_pd(_mm256_mul_pd(fp_pd, x_pd), f_pd);
        f_pd = _mm256_add_pd(_mm256_mul_pd(f_pd, x_pd), i_pd);
    }
    let mut f2 = [C0; 2];
    let mut fp2 = [C0; 2];
    unsafe {
        _mm256_storeu_pd(addr_of_mut!(f2) as *mut f64, f_pd);
        _mm256_storeu_pd(addr_of_mut!(fp2) as *mut f64, fp_pd);
    }
    Res(f2[0] + f2[1] * x, 2.0 * x * (fp2[0] + x * fp2[1]) + f2[1])
}
#[inline]
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
fn poly2_avx_fma(coe: &[Complex<f64>], x: f64) -> Res<Complex<f64>> {
    let coe_ptr = coe.as_ptr() as *const f64;
    let x_pd = _mm256_set1_pd(x * x);
    let mut f_pd = select_unpredictable(coe.len() % 2 == 0, unsafe { _mm256_loadu_pd(addr_of!(coe[coe.len() - 2]) as *const f64) }, unsafe {
        let c0 = C0;
        _mm256_loadu2_m128d(addr_of!(c0) as *const f64, addr_of!(coe[coe.len() - 1]) as *const f64)
    });
    let offset = select_unpredictable(coe.len() % 2 == 0, 2 * (coe.len() as isize - 2), 2 * (coe.len() as isize - 1));
    // if coe.len() % 2 == 0 {
    //     f_pd = unsafe { _mm256_loadu_pd(addr_of!(coe[coe.len() - 2]) as *const f64) };

    // } else {
    //     f_pd = unsafe { let c0 = C0;_mm256_loadu2_m128d(addr_of!(c0) as *const f64, addr_of!(coe[coe.len() - 1]) as *const f64) };
    // }
    let mut fp_pd = _mm256_set1_pd(0.0);
    for i in 1..coe.len().div_ceil(2) {
        let i_pd = unsafe { _mm256_loadu_pd(coe_ptr.offset(offset - i as isize * 4)) };
        fp_pd = _mm256_fmadd_pd(fp_pd, x_pd, f_pd);
        f_pd = _mm256_fmadd_pd(f_pd, x_pd, i_pd);
    }
    let mut f2 = [C0; 2];
    let mut fp2 = [C0; 2];
    unsafe {
        _mm256_storeu_pd(addr_of_mut!(f2) as *mut f64, f_pd);
        _mm256_storeu_pd(addr_of_mut!(fp2) as *mut f64, fp_pd);
    }
    Res(f2[0] + f2[1] * x, 2.0 * x * (fp2[0] + x * fp2[1]) + f2[1])
}
#[inline]
fn poly3(coe: &[Complex<f64>], x: f64) -> ResN<3, Complex<f64>> {
    let (mut f, mut fp, mut fpp) = (C0, C0, C0);

    for &i in coe.iter().skip(2).rev() {
        f.re = f.re.mul_add(x, i.re);
        f.im = f.im.mul_add(x, i.im);

        fp.re = fp.re.mul_add(x, f.re);
        fp.im = fp.im.mul_add(x, f.im);

        fpp.re = fpp.re.mul_add(x, fp.re);
        fpp.im = fpp.im.mul_add(x, fp.im);
    }
    f = x * f + coe[1];
    fp = x * fp + f;
    f = x * f + coe[0];
    fpp *= 2.0;
    ResN([f, fp, fpp])
}
#[inline]
#[target_feature(enable = "sse2")]
fn poly3_sse2(coe: &[Complex<f64>], x: f64) -> ResN<3, Complex<f64>> {
    let (mut f, mut fp, mut fpp) = (C0, C0, C0);

    let (x_pd, mut f_pd, mut fp_pd, mut fpp_pd) = (_mm_set1_pd(x), _mm_set1_pd(0.0), _mm_set1_pd(0.0), _mm_set1_pd(0.0));

    for &i in coe.iter().skip(2).rev() {
        let i_pd = unsafe { _mm_loadu_pd(addr_of!(i) as *const f64) };
        f_pd = _mm_add_pd(_mm_mul_pd(f_pd, x_pd), i_pd);
        fp_pd = _mm_add_pd(_mm_mul_pd(fp_pd, x_pd), f_pd);
        fpp_pd = _mm_add_pd(_mm_mul_pd(fpp_pd, x_pd), fp_pd);
    }
    unsafe {
        _mm_storeu_pd(addr_of_mut!(f) as *mut f64, f_pd);
        _mm_storeu_pd(addr_of_mut!(fp) as *mut f64, fp_pd);
        _mm_storeu_pd(addr_of_mut!(fpp) as *mut f64, fpp_pd);
    }
    f = x * f + coe[1];
    fp = x * fp + f;
    f = x * f + coe[0];
    fpp *= 2.0;
    ResN([f, fp, fpp])
}
#[inline]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "fma")]
fn poly3_sse2_fma(coe: &[Complex<f64>], x: f64) -> ResN<3, Complex<f64>> {
    let (mut f, mut fp, mut fpp) = (C0, C0, C0);

    let (x_pd, mut f_pd, mut fp_pd, mut fpp_pd) = (_mm_set1_pd(x), _mm_set1_pd(0.0), _mm_set1_pd(0.0), _mm_set1_pd(0.0));

    for &i in coe.iter().skip(2).rev() {
        let i_pd = unsafe { _mm_loadu_pd(addr_of!(i) as *const f64) };
        f_pd = _mm_fmadd_pd(f_pd, x_pd, i_pd);
        fp_pd = _mm_fmadd_pd(fp_pd, x_pd, f_pd);
        fpp_pd = _mm_fmadd_pd(fpp_pd, x_pd, fp_pd);
    }
    unsafe {
        _mm_storeu_pd(addr_of_mut!(f) as *mut f64, f_pd);
        _mm_storeu_pd(addr_of_mut!(fp) as *mut f64, fp_pd);
        _mm_storeu_pd(addr_of_mut!(fpp) as *mut f64, fpp_pd);
    }
    f = x * f + coe[1];
    fp = x * fp + f;
    f = x * f + coe[0];
    fpp *= 2.0;
    ResN([f, fp, fpp])
}
#[inline]
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
fn poly3_avx_fma(coe: &[Complex<f64>], x: f64) -> ResN<3, Complex<f64>> {
    let coe_ptr = coe.as_ptr() as *const f64;
    let n = coe.len();
    let x_pd = _mm256_set1_pd(x * x);

    let offset = select_unpredictable(coe.len() % 2 == 0, n as isize * 2 - 4, n as isize * 2 - 2);
    let i_pd = select_unpredictable(coe.len() % 2 == 0, unsafe { _mm256_loadu_pd(addr_of!(coe[coe.len() - 2]) as *const f64) }, unsafe {
        let c0 = C0;
        _mm256_loadu2_m128d(addr_of!(c0) as *const f64, addr_of!(coe[coe.len() - 1]) as *const f64)
    });
    // if coe.len() % 2 == 0 {
    //     i_pd = unsafe { _mm256_loadu_pd(addr_of!(coe[coe.len() - 2]) as *const f64) };
    // } else {

    //     i_pd = unsafe { let c0 = C0;_mm256_loadu2_m128d(addr_of!(c0) as *const f64, addr_of!(coe[coe.len() - 1]) as *const f64) };
    // };

    let mut f_pd = i_pd;
    let mut fp_pd = i_pd;
    let mut fpp_pd = i_pd;

    for i in 1..coe.len().div_ceil(2) - 2 {
        let i_pd = unsafe { _mm256_loadu_pd(coe_ptr.offset(offset - i as isize * 4)) };
        f_pd = _mm256_fmadd_pd(f_pd, x_pd, i_pd);
        fp_pd = _mm256_fmadd_pd(fp_pd, x_pd, f_pd);
        fpp_pd = _mm256_fmadd_pd(fpp_pd, x_pd, fp_pd);
    }

    let i_pd0 = unsafe { _mm256_loadu_pd(coe_ptr) };
    let i_pd1 = unsafe { _mm256_loadu_pd(coe_ptr.offset(4)) };
    f_pd = _mm256_fmadd_pd(f_pd, x_pd, i_pd1);
    fp_pd = _mm256_fmadd_pd(fp_pd, x_pd, f_pd);
    f_pd = _mm256_fmadd_pd(f_pd, x_pd, i_pd0);
    fpp_pd = _mm256_mul_pd(fpp_pd, _mm256_set1_pd(2.0));

    let mut f2 = [C0; 2];
    let mut fp2 = [C0; 2];
    let mut fpp2 = [C0; 2];
    unsafe {
        _mm256_storeu_pd(addr_of_mut!(f2) as *mut f64, f_pd);
        _mm256_storeu_pd(addr_of_mut!(fp2) as *mut f64, fp_pd);
        _mm256_storeu_pd(addr_of_mut!(fpp2) as *mut f64, fpp_pd);
    }
    ResN([f2[0] + f2[1] * x, 2.0 * x * (fp2[0] + x * fp2[1]) + f2[1], 2.0 * (fp2[0] + x * (2.0 * x * (fpp2[0] + x * fpp2[1]) + 3.0 * fp2[1]))])
}
#[inline]
#[target_feature(enable = "avx")]
fn poly3_avx(coe: &[Complex<f64>], x: f64) -> ResN<3, Complex<f64>> {
    let coe_ptr = coe.as_ptr() as *const f64;
    let n = coe.len();
    let x_pd = _mm256_set1_pd(x * x);

    let offset;
    let i_pd;
    if coe.len() % 2 == 0 {
        offset = n as isize * 2 - 4;
        i_pd = unsafe { _mm256_loadu_pd(addr_of!(coe[coe.len() - 2]) as *const f64) };
    } else {
        offset = n as isize * 2 - 2;
        let c0 = C0;
        i_pd = unsafe { _mm256_loadu2_m128d(addr_of!(c0) as *const f64, addr_of!(coe[coe.len() - 1]) as *const f64) };
    };

    let mut f_pd = i_pd;
    let mut fp_pd = i_pd;
    let mut fpp_pd = i_pd;

    for i in 1..coe.len().div_ceil(2) - 2 {
        let i_pd = unsafe { _mm256_loadu_pd(coe_ptr.offset(offset - i as isize * 4)) };
        f_pd = _mm256_add_pd(_mm256_mul_pd(f_pd, x_pd), i_pd);
        fp_pd = _mm256_add_pd(_mm256_mul_pd(fp_pd, x_pd), f_pd);
        fpp_pd = _mm256_add_pd(_mm256_mul_pd(fpp_pd, x_pd), fp_pd);
    }

    let i_pd0 = unsafe { _mm256_loadu_pd(coe_ptr) };
    let i_pd1 = unsafe { _mm256_loadu_pd(coe_ptr.offset(4)) };
    f_pd = _mm256_add_pd(_mm256_mul_pd(f_pd, x_pd), i_pd1);
    fp_pd = _mm256_add_pd(_mm256_mul_pd(fp_pd, x_pd), f_pd);
    f_pd = _mm256_add_pd(_mm256_mul_pd(f_pd, x_pd), i_pd0);
    fpp_pd = _mm256_mul_pd(fpp_pd, _mm256_set1_pd(2.0));

    let mut f2 = [C0; 2];
    let mut fp2 = [C0; 2];
    let mut fpp2 = [C0; 2];
    unsafe {
        _mm256_storeu_pd(addr_of_mut!(f2) as *mut f64, f_pd);
        _mm256_storeu_pd(addr_of_mut!(fp2) as *mut f64, fp_pd);
        _mm256_storeu_pd(addr_of_mut!(fpp2) as *mut f64, fpp_pd);
    }
    ResN([f2[0] + f2[1] * x, 2.0 * x * (fp2[0] + x * fp2[1]) + f2[1], 2.0 * (fp2[0] + x * (2.0 * x * (fpp2[0] + x * fpp2[1]) + 3.0 * fp2[1]))])
}
///remove the non-contributing terms from the polynomial
pub trait PolyCheck<T> {
    type Out;
    fn check_res(&mut self, x: T, err: f64);
    fn _check_res_with_init(&mut self, x: T, init: usize, err: f64);//useless (#>_<)
}
impl PolyCheck<f64> for Vec<Complex<f64>> {
    type Out = Complex<f64>;
    #[inline]
    fn check_res(&mut self, x: f64, err: f64) {
        let mut xn = x.powi((self.len() - 1) as i32);
        for (n, &i) in self.iter().enumerate().skip(1).rev() {
            let vc = xn * i;
            xn /= x;
            let dvc = xn * i * n as f64;
            if dvc.re.abs() > err || dvc.im.abs() > err || vc.re.abs() > err || vc.im.abs() > err {
                self.truncate(n + 1);
                return;
            }
        }
        unreachable!("check_res fail, the polynomial less than {} at {}", err, x);
    }
    #[inline]
    fn _check_res_with_init(&mut self, x: f64, init: usize, err: f64) {
        let err=10.0*err;
        let mut xn = x.powi((self.len() - init - 1) as i32);
        for (n, &i) in self[init..].iter().enumerate().skip(1).rev() {
            let vc = xn * i;
            xn /= x;
            let dvc = xn * i * n as f64;
            if dvc.re.abs() > err || dvc.im.abs() > err || vc.re.abs() > err || vc.im.abs() > err {
                self.truncate(n + init + 1);
                return;
            }
        }
        unreachable!("check_res_with_init fail, the polynomial less than {} at {}", err, x);
    }
}
#[test]
fn poly() {
    use crate::const_value::I;
    use crate::utilies::NUM;

    let mut coe = vec![1.2 + 0.1 * I, 1.3 + 0.1 * I, 1.4 + 0.1 * I, 1.5 + 0.1 * I, 1.6 * I, 1.7 * I, 1.8 * I, 1.14 + 1.9 * I, 5.14 + 2.0 * I];

    let v = coe.v(2.);
    let dv = coe.dvdx(2.);
    let res = coe.res(2.);
    coe.check_res(2., 1e-15);
    let res_checked = coe.res(2.);

    assert!((res.0 - v).abs() < 1e-10);
    assert!((res.1 - dv).abs() < 1e-10);
    assert!((res_checked.0 - v).abs() < 1e-10);
    assert!((res_checked.1 - dv).abs() < 1e-10);

    let old_size = coe.len();
    let res = coe.res(1e-5);
    coe.check_res(1e-5, 1e-15);
    let res_checked = coe.res(1e-5);

    assert!(coe.len() < old_size);
    assert!((res_checked.0 - res.0).abs() < 1e-10);
    assert!((res_checked.1 - res.1).abs() < 1e-10);
}
