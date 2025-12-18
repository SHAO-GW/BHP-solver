# BHP-solver
An efficient codes for solving the homogeneous Teukolsky equations

# Install

This code is writen in Rust, the minimal support version is 1.90. The calculation of the spheriodal harmonic depend on the intel MKL or LAPACKE.

# Useage

The Teukolsky equation calculted here take the following form

$`\Delta^2\frac{d}{dr}\left(\frac{1}{\Delta}\frac{dR_{lm\omega}}{dr}\right)-V(r)R_{lm\omega}=0`$

$`V(r)=\lambda-\frac{K^2-2is(r-1)K}{\Delta}-4is\omega r`$

where $`\Delta=r^2-2r+a^2`$，$`K=(r^2+a^2)\omega-ma`$.

Examples can be find [here](exsmples/error_ploter.rs)

# Reference

Please cite arxiv: 2507.15363 if you use this code.
