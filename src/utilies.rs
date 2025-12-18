use crate::const_value::{C0, C1, FNLIST, LN2_HI, LN2_LO};
use num_complex::{Complex, ComplexFloat};

use std::{
    array::from_fn,
    f64::{self, consts::LOG2_E},
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// the value and the derivative value of a function.
///
/// currently, T can only take as Complex<f64> or f64
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Res<T: Sized>(pub T, pub T);
#[inline]
pub fn exp_res(a: Res<Complex<f64>>) -> Res<Complex<f64>> {
    Res(a.0.exp(), a.0.exp() * a.1)
}
#[inline]
pub fn expmul_res(a: Res<Complex<f64>>, b: Res<Complex<f64>>) -> Res<Complex<f64>> {
    Res(a.0.exp() * b.0, a.0.exp() * (a.1 * b.0 + b.1))
}
impl<T: Display> Display for Res<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v={},\n dv={}", self.0, self.1)
    }
}
impl Add for Res<Complex<f64>> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Res(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl AddAssign for Res<Complex<f64>> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl AddAssign for Res<f64> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl Mul<Res<Complex<f64>>> for Res<Complex<f64>> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: Res<Complex<f64>>) -> Self::Output {
        Res(self.0 * rhs.0, rhs.0 * self.1 + rhs.1 * self.0)
    }
}
impl Mul<Res<f64>> for Res<Complex<f64>> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: Res<f64>) -> Self::Output {
        Res(self.0 * rhs.0, rhs.0 * self.1 + rhs.1 * self.0)
    }
}
impl Mul<Res<Complex<f64>>> for Complex<f64> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: Res<Complex<f64>>) -> Self::Output {
        Res(rhs.0 * self, rhs.1 * self)
    }
}
impl Mul<Complex<f64>> for Res<Complex<f64>> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        Res(self.0 * rhs, self.1 * rhs)
    }
}
impl Mul<f64> for Res<f64> {
    type Output = Res<f64>;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Res(self.0 * rhs, self.1 * rhs)
    }
}
impl Mul<f64> for Res<Complex<f64>> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Res(self.0 * rhs, self.1 * rhs)
    }
}
impl Mul<Res<f64>> for Res<f64> {
    type Output = Res<f64>;
    #[inline]
    fn mul(self, rhs: Res<f64>) -> Self::Output {
        Res(self.0 * rhs.0, rhs.0 * self.1 + rhs.1 * self.0)
    }
}
impl Mul<Res<Complex<f64>>> for Res<f64> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: Res<Complex<f64>>) -> Self::Output {
        Res(self.0 * rhs.0, rhs.0 * self.1 + rhs.1 * self.0)
    }
}
impl Mul<Res<Complex<f64>>> for NormResC64 {
    type Output = NormResC64;
    #[inline]
    fn mul(self, rhs: Res<Complex<f64>>) -> Self::Output {
        NormResC64 { exp2: self.exp2, res: self.res * rhs }
    }
}
impl Mul<Complex<f64>> for Res<f64> {
    type Output = Res<Complex<f64>>;
    #[inline]
    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        Res(self.0 * rhs, self.1 * rhs)
    }
}
impl MulAssign<Res<Complex<f64>>> for Res<Complex<f64>> {
    #[inline]
    fn mul_assign(&mut self, rhs: Res<Complex<f64>>) {
        // Res(self.0 * rhs.0, rhs.0 * self.1 + rhs.1 * self.0)
        self.1 *= rhs.0;
        self.1 += self.0 * rhs.1;
        self.0 *= rhs.0;
    }
}
impl Div<Complex<f64>> for Res<Complex<f64>> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Complex<f64>) -> Self::Output {
        Res(self.0 / rhs, self.1 / rhs)
    }
}
impl Div<f64> for Res<Complex<f64>> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Res(self.0 / rhs, self.1 / rhs)
    }
}
impl Div<f64> for Res<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Res(self.0 / rhs, self.1 / rhs)
    }
}
impl Sub for Res<Complex<f64>> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Res(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl Default for Res<Complex<f64>> {
    #[inline]
    fn default() -> Self {
        Res(C0, C1)
    }
}
impl<T> Res<T> {
    #[inline]
    pub const fn new(v: T, dv: T) -> Self {
        Res(v, dv)
    }
}

#[inline]
fn mask(a: f64) -> i64 {
    if a == 0.0 {
        return i64::MIN;
    }
    let mut a = a.to_bits();
    a <<= 1;
    a >>= 53;
    a as i64 - 1023
}
#[test]
fn mask_test() {
    let a = 0.25;
    print!("{}", 2.0_f64.powi(i32::MIN));
    assert_eq!(mask(a), -2);
    assert_eq!(mask(a * 2.0_f64.powi(16)), 14);
    assert_eq!(mask(a * 2.0_f64.powi(-16)), -18);
}
/// avoid overflows or underflows by an extra i64.
///
/// can represent floating-point numbers from 2^9223372036854775807 ~ -2^9223372036854775807
#[derive(Clone, Debug, Default)]
pub struct NormResC64 {
    pub exp2: i64,
    pub res: Res<Complex<f64>>,
}
impl Mul<NormResC64> for NormResC64 {
    type Output = NormResC64;
    #[inline]
    fn mul(self, rhs: NormResC64) -> Self::Output {
        Self { exp2: self.exp2 + rhs.exp2, res: self.res * rhs.res }
    }
}
impl MulAssign<Res<Complex<f64>>> for NormResC64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Res<Complex<f64>>) {
        self.res *= rhs;
    }
}
impl MulAssign<&NormResC64> for NormResC64 {
    #[inline]
    fn mul_assign(&mut self, rhs: &NormResC64) {
        self.exp2 += rhs.exp2;
        self.res *= rhs.res;
    }
}
impl DivAssign<Complex<f64>> for NormResC64 {
    #[inline]
    fn div_assign(&mut self, rhs: Complex<f64>) {
        self.res.0 /= rhs;
        self.res.1 /= rhs;
    }
}
impl NormResC64 {
    #[inline]
    pub fn from_nc2(q: &[NormC64; 2]) -> Self {
        let &[q1, q2] = q;
        if q1.exp2 > q2.exp2 { Self { exp2: q1.exp2, res: Res(q1.v, q2.v * 2.0_f64.powi((q2.exp2 - q1.exp2) as i32)) } } else { Self { exp2: q2.exp2, res: Res(q1.v * 2.0_f64.powi((q1.exp2 - q2.exp2) as i32), q2.v) } }
    }
    /// return `Res(fx.exp(),fx.exp()*dx).normalize()`
    #[inline]
    pub fn from_expfx(fx: Complex<f64>, dfx: Complex<f64>) -> Self {
        let exp2 = fx.re * LOG2_E;
        let exp2_int = exp2.round();
        let exp_rem = exp2_int.mul_add(-LN2_LO, exp2_int.mul_add(-LN2_HI, fx.re));
        let sc = fx.im.sin_cos();
        let head = (exp_rem.exp()) * Complex::new(sc.1, sc.0);
        let dhead = head * dfx;

        NormResC64 { exp2: exp2_int as i64, res: Res(head, dhead) }
    }
    #[inline]
    pub fn v(&self) -> NormC64 {
        NormC64 { exp2: self.exp2, v: self.res.0 }
    }
    #[inline]
    pub fn dv(&self) -> NormC64 {
        NormC64 { exp2: self.exp2, v: self.res.1 }
    }
    /// normalize a `NormRes` by extracting a factor `2.0_f64.powi(n)` and `self.exp2+=n`
    ///
    /// make `1<self.res.0.re.abs().max(self.res.0.im.abs()).max(self.res.1.re.abs()).max(self.res.1.im.abs())<2`
    #[inline]
    pub fn normalize(&mut self) {
        if self.res.0 == C0 {
            let exp2 = mask(self.res.1.re).max(mask(self.res.1.im));
            self.res.1 *= 2.0_f64.powi(-exp2 as i32);
            self.exp2 += exp2;
        }
        let exp2 = mask(self.res.0.re).max(mask(self.res.0.im));
        self.res.0 *= 2.0_f64.powi(-exp2 as i32);
        self.res.1 *= 2.0_f64.powi(-exp2 as i32);
        self.exp2 += exp2;
    }
    #[inline]
    pub fn to_res(self) -> Res<Complex<f64>> {
        self.res * 2.0_f64.powi(self.exp2 as i32)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct NormC64 {
    pub exp2: i64,
    pub v: Complex<f64>,
}
impl Display for NormC64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "exp^{}*({})", self.exp2, self.v)
    }
}
impl NormC64 {
    #[inline]
    pub fn to_c64(self) -> Complex<f64> {
        self.v * 2.0_f64.powi(self.exp2 as i32)
    }
    #[inline]
    pub fn recip(self) -> Self {
        Self { exp2: -self.exp2, v: NUM::recip(self.v) }
    }
    #[inline]
    pub fn from_expfx(fx: Complex<f64>) -> Self {
        let exp2 = fx.re * LOG2_E;
        let exp2_int = exp2.round();
        let exp_rem = exp2_int.mul_add(-LN2_LO, exp2_int.mul_add(-LN2_HI, fx.re));
        let sc = fx.im.sin_cos();
        let v = exp_rem.exp() * Complex::new(sc.1, sc.0);

        Self { exp2: exp2_int as i64, v }
    }
    #[inline]
    pub fn from_apowb(a: f64, b: Complex<f64>) -> Self {
        let aln = a.ln();
        Self::from_expfx(aln * b)
    }
}
#[test]
fn apowb_test() {
    use crate::const_value::I;
    let a = 2.69;
    let b = 5.3691 - 3.4886 * I;
    let c1 = NormC64::from_apowb(a, b).to_c64();
    let c2 = NUM::powc(a, b);
    println!("{}", c1 - c2)
}
impl Add<NormC64> for NormC64 {
    type Output = NormC64;
    fn add(self, rhs: NormC64) -> Self::Output {
        if self.exp2 > rhs.exp2 {
            let exp2 = rhs.exp2 - self.exp2;
            NormC64 { exp2: self.exp2, v: self.v + 2.0_f64.powi(exp2 as i32) * rhs.v }
        } else {
            let exp2 = self.exp2 - rhs.exp2;
            NormC64 { exp2: rhs.exp2, v: 2.0_f64.powi(exp2 as i32) * self.v + rhs.v }
        }
    }
}
impl Div<Complex<f64>> for NormC64 {
    type Output = NormC64;
    fn div(self, rhs: Complex<f64>) -> Self::Output {
        Self { exp2: self.exp2, v: self.v / rhs }
    }
}
impl Div<f64> for NormC64 {
    type Output = NormC64;
    fn div(self, rhs: f64) -> Self::Output {
        Self { exp2: self.exp2, v: self.v / rhs }
    }
}
impl Div<NormC64> for NormC64 {
    type Output = NormC64;
    fn div(self, rhs: NormC64) -> Self::Output {
        Self { exp2: self.exp2 - rhs.exp2, v: self.v / rhs.v }
    }
}
impl Mul<f64> for NormC64 {
    type Output = NormC64;
    fn mul(self, rhs: f64) -> Self::Output {
        Self { exp2: self.exp2, v: self.v * rhs }
    }
}
impl Mul<Complex<f64>> for NormC64 {
    type Output = NormC64;
    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        Self { exp2: self.exp2, v: self.v * rhs }
    }
}
impl Mul<NormC64> for NormC64 {
    type Output = NormC64;
    fn mul(self, rhs: NormC64) -> Self::Output {
        Self { exp2: self.exp2 + rhs.exp2, v: self.v * rhs.v }
    }
}
pub trait Zero {
    fn zero() -> Self;
}
impl Zero for Complex<f64> {
    fn zero() -> Self {
        Complex { re: 0., im: 0. }
    }
}
impl Zero for f64 {
    fn zero() -> Self {
        0.
    }
}
impl<T: Zero> Res<T> {
    pub fn zero() -> Self {
        Res(T::zero(), T::zero())
    }
}
/// the derivative value for some function, from 0th to (N-1)th,
/// N is represent the length, not the highest order of derivative values.
///
/// So, if you want nth result, you should take N=n+1 :)
#[derive(Clone, Debug)]
pub struct ResN<const N: usize, T>(pub [T; N]);
pub trait HighD {
    type T;
    type R<const N: usize>;
    fn high_d_rec<const N: usize>(&self, position: Self::T) -> Self::R<N>;
}
pub trait MutHighD {
    type T;
    type R<const N: usize>;
    fn high_d_rec<const N: usize>(&mut self, position: Self::T) -> Self::R<N>;
}
impl<const N: usize, T> Index<usize> for ResN<N, T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}
impl<const N: usize, T> IndexMut<usize> for ResN<N, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
impl<const N: usize, T> ResN<N, T>
where
    T: Default,
{
    pub fn new(v: T, dv: T) -> Self {
        let mut inner: [T; N] = from_fn(|_| T::default());
        inner[0] = v;
        inner[1] = dv;
        ResN(inner)
    }
    pub fn new_from_res(res: Res<T>) -> Self {
        let mut inner = from_fn(|_| T::default());
        inner[0] = res.0;
        inner[1] = res.1;
        ResN(inner)
    }
}
impl<T> Display for ResN<3, T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(v={}, dv={}, ddv={})", self.0[0], self.0[1], self.0[2])
    }
}
impl ResN<3, Complex<f64>> {
    pub fn exp_fx(fx: Complex<f64>, dfx: Complex<f64>, ddfx: Complex<f64>) -> Self {
        let v = fx.exp();
        let dv = v * dfx;
        let ddv = v * (NUM::powi(dfx, 2) + ddfx);

        ResN([v, dv, ddv])
    }
}
impl MulAssign<&ResN<3, Complex<f64>>> for ResN<3, Complex<f64>> {
    #[inline]
    fn mul_assign(&mut self, rhs: &ResN<3, Complex<f64>>) {
        let self1 = self[1];
        let self0 = self[0];
        self[2] *= rhs[0];
        self[2] += 2.0 * rhs[1] * self1 + self0 * rhs[2];
        self[1] *= rhs[0];
        self[1] += rhs[1] * self0;
        self[0] *= rhs[0];
    }
}
/// A fast but not arccurte version `x.powf(1./n as f64)`
///
/// When `0<n<100`, the relative error smaller than 8%.
#[inline]
pub fn fast_rootn_approx(x: f64, n: i32) -> f64 {
    if x > 1.0 {
        return 1.0 / fast_rootn_approx(1.0 / x, n);
    }

    //x=2^exp x0, x0>1&&x0<2
    //x^(1/n)=2^(exp/n)2^(exp%n as f64/n as f64) x0^(1/n)
    //x0^(1/n)=1+(x0-1)/n+O((x0-1)^2)
    let bits = x.to_bits();

    const EXP_MASK: u64 = 0x7FF0000000000000;
    const MAN_MASK: u64 = 0x000FFFFFFFFFFFFF;

    //ieee754
    //x=2^(exponent-1023)*(1+mantissa*2^-52)
    let exponent = ((bits & EXP_MASK) >> 52) as i32;
    let mantissa = bits & MAN_MASK;

    if exponent == 0 {
        return 0.0;
    }
    let exp_val = exponent - 1023;
    let new_exp = exp_val / n;
    let rem_exp = (exp_val % n) as f64 * FNLIST[n as usize];

    //piecewise linear fitting of 2^(1/rem_exp), -1<rem_exp<0
    let tail = if rem_exp > -0.5 { 1.0 + 0.585_786_437_626_905 * rem_exp } else { 0.914_213_562_373_095 + 0.414_213_562_373_095 * rem_exp };

    // let n_f64 = n as f64;
    // x0^(1/n)=1+(x0-1)/n=1+mantissa/n*2^-52
    let man_factor = 1.0 + ((mantissa / n as u64) as f64) * 2.0_f64.powi(-52);

    man_factor * 2.0_f64.powi(new_exp) * tail
}
#[test]
fn fast_rootn_error_check() {
    use fastrand::f64;
    let mut max_err = 0.0;
    for n in 3..100 {
        for _ in 0..10000 {
            let x = f64();
            let err = ((fast_rootn_approx(x, n) - x.powf(1.0 / n as f64)) / x.powf(1.0 / n as f64)).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    for n in 3..100 {
        for _ in 0..10000 {
            let x = f64() * 1e6;
            let err = ((fast_rootn_approx(x, n) - x.powf(1.0 / n as f64)) / x.powf(1.0 / n as f64)).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    assert!(max_err < 0.08);
}
///some basic numerical function for f64 or Complex<f64>
pub trait NUM {
    ///Computes the absolute value of self
    fn abs(self) -> f64;
    ///the exponential function
    fn exp(self) -> Self;
    ///Raises a number to an integer power.
    fn powi(self, n: i32) -> Self;
    ///Takes the reciprocal (inverse) of a number, 1/x.
    fn recip(self) -> Self;
    ///Raises self to a complex power.
    fn powc(self, exp: Complex<f64>) -> Complex<f64>;
}

impl NUM for Complex<f64> {
    #[inline]
    fn abs(self) -> f64 {
        //copy from https://github.com/rust-lang/compiler-builtins/blob/master/libm/src/math/hypot.rs
        //this function only used for error estimate, so its accuracy doesn't matter.
        //i also delete some check for inf or nan :)
        let Complex { re: mut x, im: mut y } = self;
        let x1p700 = f64::from_bits(0x6bb0000000000000); // 0x1p700 === 2 ^ 700
        let x1p_700 = f64::from_bits(0x1430000000000000); // 0x1p-700 === 2 ^ -700

        let mut uxi = x.to_bits();
        let mut uyi = y.to_bits();
        let mut z: f64;

        /* arrange |x| >= |y| */
        uxi &= -1i64 as u64 >> 1;
        uyi &= -1i64 as u64 >> 1;
        if uxi < uyi {
            std::mem::swap(&mut uxi, &mut uyi);
        }

        /* special cases */
        let ex = (uxi >> 52) as i64;
        let ey = (uyi >> 52) as i64;
        x = f64::from_bits(uxi);
        y = f64::from_bits(uyi);

        /* note: hypot(x,y) ~= x + y*y/x/2 with inexact for small y/x */
        /* 64 difference is enough for ld80 double_t */
        if ex - ey > 64 {
            return x + y;
        }
        z = 1.;
        if ex > 0x3ff + 510 {
            z = x1p700;
            x *= x1p_700;
            y *= x1p_700;
        } else if ey < 0x3ff - 450 {
            z = x1p_700;
            x *= x1p700;
            y *= x1p700;
        }
        z * (x * x + y * y).sqrt()
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        ComplexFloat::powi(self, n)
    }
    /// a more robost `recip` than ComplexFloat::recip
    #[inline]
    fn recip(self) -> Self {
        if self.im.abs() >= self.re.abs() {
            let r = self.re / self.im;
            let t = (self.im + self.re * r).recip();

            if r != 0.0 { Complex { re: r * t, im: -t } } else { Complex { re: 0., im: -t } }
        } else {
            let r = self.im / self.re;
            let t = (self.re + self.im * r).recip();

            if r != 0.0 { Complex { re: t, im: -r * t } } else { Complex { re: t, im: 0. } }
        }
        // ComplexFloat::recip(self)
    }
    #[inline]
    fn powc(self, exp: Complex<f64>) -> Complex<f64> {
        self.powc(exp)
    }
}
impl NUM for f64 {
    #[inline]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }
    #[inline]
    fn powc(self, exp: Complex<f64>) -> Complex<f64> {
        ComplexFloat::powc(self, exp)
    }
}

#[test]
fn num_test() {
    let OV = 2.0_f64.powi(-1022);
    print!("{:?}\n{:?}", OV, 2.0 / (f64::EPSILON * f64::EPSILON))
}
///trait for f64 or Complex<f64>, should not impl for any other type.
pub trait FC64
where
    Self: NUM + Sized + Clone + Copy + AddAssign<Self> + SubAssign<Self> + Neg<Output = Self> + Mul<Self, Output = Self> + Sub<Self, Output = Self> + Add<Self, Output = Self> + Div<Self, Output = Self> + Mul<Complex<f64>, Output = Complex<f64>> + Sub<Complex<f64>, Output = Complex<f64>> + Add<Complex<f64>, Output = Complex<f64>> + Div<Complex<f64>, Output = Complex<f64>> + Mul<f64, Output = Self> + Add<f64, Output = Self> + Sub<f64, Output = Self> + Div<f64, Output = Self> + Zero,
{
}
impl FC64 for Complex<f64> {}
impl FC64 for f64 {}

#[test]
fn abs_check() {
    let a: Complex<f64> = 2. * Complex::ONE + 2.39 * Complex::I;
    assert!((NUM::abs(a) - ComplexFloat::abs(a)).abs() < 1e-12);
    assert!((NUM::abs(a / 1e200_f64) * 1e200_f64 - NUM::abs(a)).abs() < 1e-12);
    assert!((NUM::abs(a * 1e200_f64) / 1e200_f64 - NUM::abs(a)).abs() < 1e-12);

    let a: Complex<f64> = 2.39 * Complex::ONE + 1e-200 * Complex::I;
    assert!((NUM::abs(a) - ComplexFloat::abs(a)).abs() < 1e-12);
    assert!((NUM::abs(a / 1e200_f64) * 1e200_f64 - NUM::abs(a)).abs() < 1e-12);
    assert!((NUM::abs(a * 1e200_f64) / 1e200_f64 - NUM::abs(a)).abs() < 1e-12);

    let a: Complex<f64> = 1e-200 * Complex::ONE + 2.39 * Complex::I;
    assert!((NUM::abs(a) - ComplexFloat::abs(a)).abs() < 1e-12);
    assert!((NUM::abs(a / 1e200_f64) * 1e200_f64 - NUM::abs(a)).abs() < 1e-12);
    assert!((NUM::abs(a * 1e200_f64) / 1e200_f64 - NUM::abs(a)).abs() < 1e-12);

    let a: Complex<f64> = 2.39e-200 * Complex::ONE + 1e-200 * Complex::I;
    assert!((NUM::abs(a) - ComplexFloat::abs(a)).abs() * 1e200 < 1e-12);
    assert!((NUM::abs(a * 1e200_f64) / 1e200_f64 - NUM::abs(a)).abs() * 1e200 < 1e-12);
}

#[test]
fn recip_check() {
    let a: Complex<f64> = 2.39 * Complex::ONE + 2. * Complex::I;
    assert!(NUM::abs(NUM::recip(a) - ComplexFloat::recip(a)) < 1e-12);
    assert!(NUM::abs(NUM::recip(a / 1e200_f64) / 1e200_f64 - NUM::recip(a)) < 1e-12);
    assert!(NUM::abs(NUM::recip(a * 1e200_f64) * 1e200_f64 - NUM::recip(a)) < 1e-12);

    let a: Complex<f64> = 2.39 * Complex::ONE + 1e-200 * Complex::I;
    assert!(NUM::abs(NUM::recip(a) - ComplexFloat::recip(a)) < 1e-12);
    assert!(NUM::abs(NUM::recip(a / 1e200_f64) / 1e200_f64 - NUM::recip(a)) < 1e-12);
    assert!(NUM::abs(NUM::recip(a * 1e200_f64) * 1e200_f64 - NUM::recip(a)) < 1e-12);

    let a: Complex<f64> = 1e-200 * Complex::ONE + 2.39 * Complex::I;
    assert!(NUM::abs(NUM::recip(a) - ComplexFloat::recip(a)) < 1e-12);
    assert!(NUM::abs(NUM::recip(a / 1e200_f64) / 1e200_f64 - NUM::recip(a)) < 1e-12);
    assert!(NUM::abs(NUM::recip(a * 1e200_f64) * 1e200_f64 - NUM::recip(a)) < 1e-12);

    let a: Complex<f64> = 2.39e-200 * Complex::ONE + 1e-200 * Complex::I;
    assert!(NUM::abs(NUM::recip(a) - ComplexFloat::recip(a)) * 1e-200 < 1e-12);
}
pub trait HeapSize {
    fn get_heapsize(&self) -> usize;
}
impl<T: Sized> HeapSize for Vec<T> {
    fn get_heapsize(&self) -> usize {
        self.capacity() * size_of::<T>()
    }
}
