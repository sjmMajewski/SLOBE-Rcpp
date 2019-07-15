/*
 * Copyright 2019, Szymon Majewski, Błażej Miasojedow
 *
 * This file is part of SLOBE-Rcpp Toolbox
 *
 *   The SLOBE-Rcpp Toolbox is free software: you can redistribute it
 *   and/or  modify it under the terms of the GNU General Public License
 *   as published by the Free Software Foundation, either version 3 of
 *   the License, or (at your option) any later version.
 *
 *   The SLOPE Toolbox is distributed in the hope that it will
 *   be useful, but WITHOUT ANY WARRANTY; without even the implied
 *   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *   See the GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with the SLOPE Toolbox. If not, see
 *   <http://www.gnu.org/licenses/>.
 */

#include<RcppArmadillo.h>
#include<math.h>
#include<stdlib.h>
#include<numeric>
#include<algorithm>

//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

/*
 * Copyright 2013, M. Bogdan, E. van den Berg, W. Su, and E.J. Candes
 *
 * The following function is copied from proxSortedL1.c file 
 * which is part of SLOPE Toolbox.
 *
 *   The SLOPE Toolbox is free software: you can redistribute it
 *   and/or  modify it under the terms of the GNU General Public License
 *   as published by the Free Software Foundation, either version 3 of
 *   the License, or (at your option) any later version.
 *
 *   The SLOPE Toolbox is distributed in the hope that it will
 *   be useful, but WITHOUT ANY WARRANTY; without even the implied
 *   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *   See the GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with the SLOPE Toolbox. If not, see
 *   <http://www.gnu.org/licenses/>.
 */

int evaluateProx(double *y, double *lambda, double *x, size_t n, int *order);

/* ----------------------------------------------------------------------- */
int evaluateProx(double *y, double *lambda, double *x, size_t n, int *order)
/* ----------------------------------------------------------------------- */
{  double   d;
   double  *s     = NULL;
   double  *w     = NULL;
   size_t  *idx_i = NULL;
   size_t *idx_j = NULL;
   size_t  i,j,k;
   int      result = 0;

   /* Allocate memory */
   s     = (double *)malloc(sizeof(double) * n);
   w     = (double *)malloc(sizeof(double) * n);
   idx_i = (size_t *)malloc(sizeof(size_t) * n);
   idx_j = (size_t *)malloc(sizeof(size_t) * n);

   if ((s != NULL) && (w != NULL) && (idx_i != NULL) && (idx_j != NULL))
   {
      k = 0;
      for (i = 0; i < n; i++)
      {
         idx_i[k] = i;
         idx_j[k] = i;
         s[k]     = y[i] - lambda[i];
         w[k]     = s[k];

         while ((k > 0) && (w[k-1] <= w[k]))
         {  k --;
            idx_j[k] = i;
            s[k]    += s[k+1];
            w[k]     = s[k] / (i - idx_i[k] + 1);
         }

         k++;
      }

      if (order == NULL)
      {  for (j = 0; j < k; j++)
         {  d = w[j]; if (d < 0) d = 0;
            for (i = idx_i[j]; i <= idx_j[j]; i++)
            {  x[i] = d;
            }
         }
      }
      else
      {  for (j = 0; j < k; j++)
         {  d = w[j]; if (d < 0) d = 0;
            for (i = idx_i[j]; i <= idx_j[j]; i++)
            {  x[order[i]] = d;
            }
         }
      }
   }
   else
   {  result = -1;
   }

   /* Deallocate memory */
   if (s     != NULL) free(s);
   if (w     != NULL) free(w);
   if (idx_i != NULL) free(idx_i);
   if (idx_j != NULL) free(idx_j);

   return result;
}

// Comparator class for argsort function
class CompareByNumericVectorValues {
	private:
		const NumericVector* _values;
  
	public:
		CompareByNumericVectorValues(const NumericVector* values) { _values = values;}
		bool operator() (const int& a, const int& b) {return ((*_values)[a] > (*_values)[b]);}
};


// Writes down in IntegerVector ord sequnce of indexes,
// such that w[ord[i]] >= w[ord[j]] whenever i <= j 
// for the given NumericVector w.
void argsort(const NumericVector& w, IntegerVector ord) {
	std::iota(ord.begin(), ord.end(), 0);
	CompareByNumericVectorValues comp = CompareByNumericVectorValues(&w);
	std::sort(ord.begin(), ord.end(), comp);
}

// Computes proximal step of SLOPE(lambda) norm from point y
// NumericVector y is not assumed to be sorted, sorting is performed within function
NumericVector prox_sorted_L1_C(NumericVector y, NumericVector lambda) {
 	size_t n = y.size();
 	NumericVector x(n);
 	IntegerVector order(n);
 	argsort(abs(y),order);
 	IntegerVector sign_y= sign(y);
 	y = abs(y);
 	y.sort(true);
 	evaluateProx(y.begin(), lambda.begin(), x.begin(), n, NULL);
 	NumericVector res(n);
	for(int k=0;k<n;k++){
		res[order[k]]= sign_y[order[k]]*x[k];
	}
	return res;
}

// Creates a vector of weights for SLOPE for a given p and FDR
// Writes down the vector in the passed NumericVector lam
void create_lambda(NumericVector& lam, int p, double FDR) {
	NumericVector h(p);
	for (double i = 0.0; i < h.size(); ++i) {
		h[i] = 1 - (FDR* (i+1)/(2*p));
	}
	lam = qnorm(h);
}

// Expectation of truncated gamma distribution
double EX_trunc_gamma(double a ,double b ){
  double c = exp(Rf_pgamma(1.0, a+1, 1.0/b, 1, 1) - Rf_pgamma(1.0, a, 1.0/b, 1, 1));
  c /= b;
  c *= a;
  return c;
}

// Compute the SLOPE estimator for a liner model using ADMM
arma::vec slope_admm(const mat& X, const vec& Y, NumericVector& lambda,
				const int& p, const double& rho, int max_iter=500, double tol_inf = 1e-08) {
	
		// Precompute M = (X^TX + rho I)^{-1} 
		// and MXtY = M * X^T * Y for proximal steps of quadratic part
		mat M = X.t() * X;
		for (int i=0; i<p; ++i) {
			M.at(i,i) += rho;
		}
		M = M.i();
		vec MXtY = M * (X.t() * Y);
		NumericVector lam_seq_rho = lambda/rho;
		
		// Prepare variables before starting ADMM loop
		int i=0;
		vec x = zeros(p);
		vec z = zeros(p);
		vec u = zeros(p);
		NumericVector z_new = NumericVector(p);
		vec z_new_arma = zeros(p);
		NumericVector x_plus_u(p);
		double dual_feas, primal_feas;
		
		// ADMM loop
		while (i < max_iter) {
			x = MXtY + M*(rho*(z - u));
			x_plus_u = as<NumericVector>(wrap(x+u));
			z_new = prox_sorted_L1_C(x_plus_u, lam_seq_rho);
			z_new_arma = as<arma::vec>(z_new);
			u += (x - z_new_arma);
			
			dual_feas = arma::norm(rho*(z_new_arma - z));
			primal_feas = arma::norm(z_new_arma - x);
			
			z = z_new_arma;
			if (primal_feas < tol_inf && dual_feas < tol_inf){
				i = max_iter;
			}

			++i;
		}
		
		return z;
}

void div_X_by_w(mat& X_div_w, const mat& X, const vec& w_vec, const int& n, const int& p) {
	for (int i=0; i<n; ++i) {
		for (int j=0; j<p; ++j) {
			X_div_w.at(i,j) = X.at(i,j) / w_vec(j);
		}
	}
}

// [[Rcpp::export]]
List SLOBE_ADMM_approx(NumericVector start, mat X, vec Y, double a_prior, double b_prior, double sigma = 1.0, 
				double FDR = 0.05, double tol = 1e-04, bool known_sigma = false, int max_iter=100, bool verbose = true) {

	// Initialize variables
	int p = start.length();
	int n = Y.size();
	NumericVector beta = clone(start);
	NumericVector beta_new(p);
	vec beta_arma = as<vec>(beta);
	NumericVector w(p, 1.0);
	vec w_vec = ones(p);
	NumericVector wbeta(p);
	NumericVector gamma(p);
	NumericVector gamma_h(p);
	NumericVector b_sum_h(p);
	NumericVector lambda_sigma(p);
	IntegerVector order(p);
	mat X_div_w = zeros(n,p);
  	double error = 0.0;
	double swlambda = 0.0;
	double RSS = 0.0;
	
	// Compute vector lambda based on BH procedure
	NumericVector lambda(p);
	create_lambda(lambda, p, FDR);

	// Initialize c, theta
	double sstart = sum(start != 0);
	double c = 0.0;
	if (sstart > 0) {
		double h = (sstart+1)/(abs(sstart * lambda[p-1] * sigma));
		c = (h < 0.9) ? h : 0.9;
	}
	else
		c = 0.9;	
  
	double theta = (sstart + a_prior)/(a_prior + b_prior + p);

	// Start main loop
	bool converged = false;
	int iter = 0;
	while (iter < max_iter) {
	  if(verbose){
		  Rcout << "Iteracja: " << iter <<"\n" ;
	  }
		wbeta = w * abs(beta);
		argsort(wbeta, order);

		// For the version with unknown sigma, compute it first
		if (!known_sigma) {
			RSS = sum(pow((X * beta_arma - Y),2));
			swlambda = sum(wbeta.sort(true) * lambda);
			sigma = (swlambda + sqrt(pow(swlambda, 2.0) + 4*n*RSS))/(2*n);
		}

		// compute new gamma
		gamma_h = abs(beta[order]) * (c-1) * lambda / sigma ;
		gamma_h = (theta * c)/(theta * c + (1-theta) * exp(gamma_h));

		// update c
		double sum_gamma = sum(gamma);
		b_sum_h = gamma[order];
		b_sum_h = b_sum_h * abs(beta[order]);
		b_sum_h = b_sum_h * lambda;
		double b_sum = sum(b_sum_h)/sigma;
		if (sum_gamma > 0) {
			if (b_sum > 0) 
				c = EX_trunc_gamma(sum_gamma, b_sum);	
			else
				c = sum_gamma/(sum_gamma + 1);
			}
		else
			c = 0.5;
	
		// update theta, gamma, w based on previous calculations
		theta = (sum_gamma + a_prior)/(p + a_prior + b_prior);
		//std::copy(gamma_h.begin(), gamma_h.end(), gamma.begin());
		for(int i=0; i<p;++i){
		  gamma[order[i]]=gamma_h[i];
		}
		w = 1.0 - (1.0 - c) * gamma;

		
		// Compute rewieghted SLOPE estimator using computed weights and sigma
		lambda_sigma = lambda * sigma;
		w_vec = as<vec>(w);
		div_X_by_w(X_div_w, X, w_vec, n, p);
		beta_arma = slope_admm(X_div_w, Y, lambda_sigma, p, 1.0);
		for (int i=0; i<p; ++i) {
			beta_arma[i] /= w_vec[i];
		}
		beta_new = as<NumericVector>(wrap(beta_arma));
		
		// Check stop condition
		error= sum(abs(beta-beta_new));
		if (error < tol) {
			iter = max_iter;
			converged = true;
		}
		if(verbose){
		
	 	  Rcout<< "Error =  "<< error <<" sigma = "<< sigma <<" theta = "<< theta<<" c = "<< c<<"\n";
		}
		std::copy(beta_new.begin(), beta_new.end(), beta.begin()) ;
		++iter;
	}


	return List::create(Named("beta")=beta, Named("sigma")=sigma, Named("theta")=theta, Named("c")=c,
                           Named("w")=w, Named("converged")=converged);
}
