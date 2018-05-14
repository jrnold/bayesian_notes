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
  // scale of HS slab
  real<lower=0.> scale_slab;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha_raw;
  vector[K] beta_raw;
  real<lower=0.> sigma;
  // hyper-parameters of coefficients
  real<lower=0.> tau;
  // local scales
  vector[K] lambda;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;
  vector[K] lambda_tilde;

  alpha = alpha_raw * scale_alpha;
  // calculate these inside a block to avoid savin them
  // calculate these values to avoid calculating them multiple times
  for (k in 1:K) {
    real lambda2;
    real c2;
    c2 = pow(scale_slab, 2);
    lambda2 = pow(lambda[k], 2);
    lambda_tilde[k] = sqrt(c2 * lambda2 ./ (lambda2 + c2));
  }
  beta = beta_raw * tau .* lambda_tilde;
  mu = alpha + X * beta;
}
model {
  // hyperpriors
  tau ~ exponential(loc_tau);
  // priors
  alpha_raw ~ normal(0., 1.);
  beta_raw ~ normal(0., 1.);
  sigma ~ exponential(loc_sigma);
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
