//!c api for series expand method

use crate::jh::*;
use num_complex::Complex;
use std::{mem::ManuallyDrop, ptr::drop_in_place};

#[repr(C)]
pub union Series {
    a: [f64; 65],
    b: ManuallyDrop<SE<f64>>,
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_new(ser: *mut Series, s: f64, l: f64, m: f64, q: f64, omega: f64, lambda: f64) {
    unsafe {
        (*ser).b = ManuallyDrop::new(SE::new(s, l, m, q, omega, lambda));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_cal(ser: *mut SE<f64>, r: f64, result: *mut [Complex<f64>; 4]) {
    unsafe {
        let ans = (*ser).cal(r);
        *result = [ans.0.0, ans.0.1, ans.1.0, ans.1.1];
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_btrans(ser: *const SE<f64>) -> Complex<f64> {
    unsafe { (*ser).btrans() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_binc(ser: *const SE<f64>) -> Complex<f64> {
    unsafe { (*ser).binc() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_bref(ser: *const SE<f64>) -> Complex<f64> {
    unsafe { (*ser).bref() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_ctrans(ser: *const SE<f64>) -> Complex<f64> {
    unsafe { (*ser).ctrans() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ser_drop(ser: *mut SE<f64>) {
    unsafe {
        drop_in_place(ser);
    }
}
#[test]
fn test() {
    use std::mem::{align_of, size_of};
    assert_eq!(align_of::<Series>(),align_of::<SE<f64>>());
    assert_eq!(size_of::<Series>(),size_of::<SE<f64>>());
    assert_eq!(size_of::<Series>(),size_of::<[f64;65]>());
}
