# SLOBE-Rcpp
Code for SLOBE algorithm in Rcpp

# Usage
Both files are meant to be imported to R using the sourcecpp function in Rcpp library:

library(Rcpp)

Rcpp::sourceCpp('SLOBE_cpp.cpp')
Rcpp::sourceCpp('SLOBE_cpp_missing.cpp')

This will create in R environment functions 'SLOBE_admm_approx' and 'SLOBE_admm_approx_missing'
