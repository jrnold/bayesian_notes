/*
  Linear Model with Spike-Slab prior.
  This probably won't work well.

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
  // Mixture prior
  real<lower=0.> shape1_theta;
  real<lower=0.> shape2_theta;
  // Student-t prior on slab
  real<lower=0.> df_slab;
  real<lower=0.> scale_slab;
  // Normal distribution on spike
  real<lower=0.> scale_spike;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression intercept
  real alpha_z;
  // regression coefficients
  vector[K] beta_z;
  // scale of regression errors
  real<lower=0.> sigma;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;

  alpha = scale_alpha * alpha_z;
  beta = tau * sqrt(lambda2) .* beta_z;
  mu = alpha + X * beta;
}
model {
  // mixture weight
  theta ~ beta(shape1_theta, shape2_theta);
  // hyperpriors
  spike_z ~ normal(0., 1.);
  slab_z ~ normal(0., 1.);
  // priors
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  sigma ~ exponential(rate_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  // shrinkage factors
  vector[K] kappa;
  // number of effective coefficients
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
    kappa = 1. ./ (1. + N * inv_sigma2 * tau2 * lambda2);
  }
  m_eff = K - sum(kappa);
}
