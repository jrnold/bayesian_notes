/*

# Linear Model with Normal Errors

This version uses a centered parameterization

*/
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // scale of normal prior on regression intercept
  real<lower=0.> scale_alpha;
  // scale of normal prior on regression coefficients
  vector<lower=0.>[K] scale_beta;
  // expected value of the regression error
  real<lower=0.> loc_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha_z;
  vector[K] beta_z;
  real<lower=0.> sigma_z;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;

  alpha = alpha_z * scale_alpha;
  beta = beta_z * scale_beta;
  mu = alpha + X * beta;
}
model {
  // priors
  alpha_z ~ normal(0., 1);
  beta_z ~ normal(0., 1.);
  sigma_z ~ exponential(1.);
  // likelihood
  y ~ normal(mu, sigma_z * loc_sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
