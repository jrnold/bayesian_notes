// Linear Model with Normal Errors
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
  // priors on alpha
  real<lower=0.> scale_alpha;
  // prior scale on global
  real<lower=0.> loc_tau;
  // prior on regression error distribution
  real<lower=0.> loc_sigma;
  // degrees of freedom for half-t prior on tau
  real<lower=1.> df_tau;
  // degrees of freedom for half-t prior on lambdas
  real<lower=1.> df_lambda;
  // scale and degrees of freedom of HS slab
  real<lower=0.> scale_slab;
  real<lower=0.> df_slab;
  // Number of non-zero coefficients expected;
  // will be multiplied by sigma.
  real<lower=0.,upper=K> k0;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
transformed data {
  real<lower=0.> tau0 = k0 / (K - k0) / sqrt(1. * N);
}
parameters {
  // regression coefficients on z-scale
  real alpha_z;
  vector[K] beta_z;
  // regression variance
  real<lower=0.> sigma;
  // global shrinkage parameter
  real<lower=0.> tau;
  // local shrinkage
  vector[K] lambda;
  // slab scale
  real<lower=0.> c;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;
  vector[K] lambda_tilde;

  alpha = alpha_z * scale_alpha;
  // calculate these inside a block to avoid savin them
  // calculate these values to avoid calculating them multiple times
  lambda_tilde = c * lambda ./ (c ^ 2 + tau ^ 2 * square(lambda));
  beta = beta_z * tau .* lambda_tilde;
  mu = alpha + X * beta;
}
model {
  // regression noise
  sigma ~ exponential(loc_sigma);
  // global shrinkage parameters
  tau ~ student_t(df_tau, 0., sigma * tau0);
  // local shrinkage parameters
  lambda ~ student_t(df_lambda, 0., 1.);
  // priors
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  // slab scale
  c ~ student_t(df_slab, 0, scale_slab);
  // likelihood
  y ~ normal(mu, sigma);
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
