/*
  Linear Model with Normal Errors and Regularized Horseshoe shrinkage
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
  // priors on alpha
  real<lower=0.> scale_alpha;
  // prior scale on global
  real<lower=0.> scale_tau;
  real<lower=1.> df_tau;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // degrees of freedom for half-t prior on lambdas
  real<lower=1.> df_lambda;
  // scale and degrees of freedom of HS slab
  real<lower=0.> scale_slab;
  real<lower=0.> df_slab;
  // flags
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
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
  vector<lower = 0.>[K] lambda;
  // slab scale
  real<lower=0.> c;
}
transformed parameters {
  // expected value of the response
  vector[N] mu;
  // regularized lambda
  vector<lower = 0.>[K] lambda_tilde;
  real alpha;
  vector[K] beta;

  alpha = alpha_z * scale_alpha;
  beta = tau * lambda .* beta_z;
  lambda_tilde = c * lambda ./ (c ^ 2 + tau ^ 2 * square(lambda));
  mu = alpha + X * beta;
}
model {
  // regression noise
  sigma ~ exponential(rate_sigma);
  // global shrinkage parameters
  tau ~ student_t(df_tau, 0., sigma * scale_tau);
  // local shrinkage parameters
  lambda ~ student_t(df_lambda, 0., 1.);
  // regression coefficient and
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
  vector[K] kappa;
  real m_eff;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
  {
    real inv_sigma2 = 1. / sigma ^ 2;
    real tau2 = tau ^ 2;
    vector[K] lambda2 = square(lambda);
    kappa = 1. ./ (1. + N * inv_sigma2 * tau2 * lambda2);
  }
  m_eff = K - sum(kappa);
}
