// Negative Binomial Distribution
data {
  // number of observations
  int<lower=0> N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0> y[N];
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  real<lower=0.> scale_beta;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  // 1 / sqrt(phi)
  real<lower=0.> inv_sqrt_phi;
}
transformed parameters {
  vector[N] eta;
  real<lower=0.> phi;

  phi = 1 / inv_sqrt_phi ^ 2;
  eta = alpha + X * beta;
}
model {
  // priors
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  inv_sqrt_phi ~ normal(0., 1.);
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  // likelihood
  y ~ neg_binomial_2_log(eta, phi);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = neg_binomial_2_rng(exp(eta[i]), phi);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | eta[i], phi);
  }
}
