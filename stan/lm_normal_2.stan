// Linear Model with Normal Errors -
// version for non-scaled and centered data
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
  real<lower=0> scale_alpha;
  real[K]<lower=0> scale_beta;
  real<lower=0> loc_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha_raw;
  vector[K] beta_raw;
  real<lower=0> sigma_raw;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta_raw;
  real<lower=0> sigma_raw;

  beta = beta_raw .* sd_X * sd_y;
  alpha = alpha_raw * sd_y;
  sigma = sigma_raw * sd_y;
  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0., 1.);
  beta ~ normal(0., 1.);
  sigma ~ exponential(1.);
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
