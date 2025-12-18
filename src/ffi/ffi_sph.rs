use crate::{
    utilies::Res,
    sph::{SphCoeF64, SphEigen, SphListEigen, Spherical, Spheroidal},
};
use num_complex::Complex;
use std::ptr::drop_in_place;

//__________________________________________________________________________
///C API///
#[repr(C)]
pub struct SphF64 {
    inner: [f64; 9],
}
#[repr(C)]
pub struct SphC64 {
    inner: [f64; 10],
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_f64_new(sph: *mut Spheroidal<f64>, s: f64, l: f64, m: f64, c: f64, delta: f64) {
    unsafe {
        sph.write(Spheroidal::new(s, l, m, c, delta));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_f64_lambda(sph: *const Spheroidal<f64>) -> f64 {
    unsafe { (*sph).lambda }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_f64_value(sph: *const Spheroidal<f64>, x: f64, res: &mut Res<f64>) {
    unsafe {
        *res = (*sph).at(x);
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_f64_drop(sph: *mut Spheroidal<f64>) {
    unsafe { drop_in_place(sph) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_c64_new(sph: *mut Spheroidal<Complex<f64>>, s: f64, l: f64, m: f64, c: Complex<f64>, delta: f64) {
    unsafe {
        sph.write(Spheroidal::new(s, l, m, c, delta));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_c64_lambda(sph: *const Spheroidal<Complex<f64>>) -> Complex<f64> {
    unsafe { (*sph).lambda }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_c64_value(sph: *const Spheroidal<Complex<f64>>, x: f64) -> Res<Complex<f64>> {
    let ans;
    unsafe { ans = (*sph).at(x) }
    ans
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sph_c64_drop(sph: *mut Spheroidal<Complex<f64>>) {
    unsafe { drop_in_place(sph) };
}

#[test]
fn test() {
    use std::mem::{align_of, size_of};
    println!("{}", align_of::<SphF64>());
    println!("{}", size_of::<Spheroidal<f64>>() / 8);
    println!("{}", size_of::<[f64; 9]>());
    println!("{}", size_of::<SphF64>());
}
//__________________________________________________________________________

// #[repr(C)]
// pub struct SphCoeF64 {
//     inner: [f64; 13],
// }
#[repr(C)]
pub struct SphericalF64 {
    inner: [f64; 11],
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spheroidal_coe_f64_new(sph: *mut SphCoeF64, s: f64, lmax: f64, m: f64, c: f64, delta: f64) {
    unsafe {
        sph.write(SphCoeF64::new(s, lmax, m, c, delta));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spheroidal_f64_cal(sph: *const SphCoeF64, spherical: *mut Spherical, l: f64, res: *mut Res<f64>) {
    unsafe {
        *res = (*sph).at(spherical.as_mut().unwrap_unchecked(), l);
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spherical_f64_cal(sph: *mut Spherical, s: f64, m: f64, x: f64) {
    unsafe {
        sph.write(Spherical::new(s, m, x));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spherical_f64_drop(sph: *mut Spherical) {
    unsafe { drop_in_place(sph) };
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spheroidal_coe_drop(sph: *mut SphCoeF64) {
    unsafe { drop_in_place(sph) };
}
#[test]
fn sph_list_size_check() {
    use std::mem::{align_of, size_of};
    println!("{}", size_of::<SphCoeF64>() / 8);
    println!("{}", size_of::<Spherical>() / 8);
    println!("{}", align_of::<SphCoeF64>() / 8);
    println!("{}", align_of::<Spherical>() / 8);
}
