//! special functions
#[cfg(test)]
use crate::const_value::LG_COE;

///the lgamma function. ```ln(Gamma(x))```
#[cfg(test)]
pub fn real_lgamma(x: f64) -> f64 {
    let mut y = x;
    let tmp = (x + 0.5) * (x + 5.242_187_5).ln() - x - 5.242_187_5;
    let mut ser = 0.999_999_999_999_997_1;
    for a in LG_COE {
        y += 1.0;
        ser += a / y;
    }
    tmp + (2.5066282746310005 * ser / x).ln()
} //W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery, Numerical Recipes (Cambridge University Press, Cambridge, 1992).

#[inline]
pub fn jacobi_poly_n(n: f64, a: f64, b: f64, x: f64, jn_1: f64, jn_2: f64) -> f64
//c0(n,a,b)j(n-2,a,b)+c1(n,a,b)j(n-1,a,b)->j(n,a,b) 迭代关系
{
    // (jn_2*(-2.*(-1.+a)*(-1.+b)*(a+b)-2.*(2.+a.powi(2)+4.*a*(-1.+b)+(-4.+b)*b)*n+(8.-6.*a-6.*b)*n.powi(2)-4.*n.powi(3))+jn_1*(12.*(-1.+a+b)*x*n.powi(2)+8.*x*n.powi(3)+(-1.+a+b)*(a+b)*(a-b+(-2.+a+b)*x)+2.*(a.powi(2)-b.powi(2)+2.*x+3.*(-2.+a+b)*(a+b)*x)*n))/(2.*n*(a+b+n)*(-2.+a+b+2.*n))
    let a = n + a;
    let b = n + b;
    let c = a + b;

    ((c - 1.0) * (c * (c - 2.0) * x + (a - b) * (c - 2.0 * n)) * jn_1 - 2.0 * (a - 1.0) * (b - 1.0) * c * jn_2) / (2.0 * n * (c - n) * (c - 2.0))
} //https://mathworld.wolfram.com/JacobiPolynomial.html
#[inline]
pub fn djacobi_poly_n(n: f64, a: f64, b: f64, x: f64, jn: f64, jn_1: f64) -> f64
//c0(n,a,b)j(n-1,a,b)+c1(n,a,b)j(n,a,b)->dj(n,a,b) 迭代关系
{
    let a = n + a;
    let b = n + b;
    let c = a + b;

    (-2.0 * a * b * jn_1 + n * (b - a + c * x) * jn) / (c * (x * x - 1.0))
} //https://mathworld.wolfram.com/JacobiPolynomial.html