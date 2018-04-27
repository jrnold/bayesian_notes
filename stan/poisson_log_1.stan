// Poisson GLM
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
}
transformed parameters {
  vector[N] eta;

  eta = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  // likelihood
  y ~ poisson_log(eta);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = poisson_rng(exp(eta[i]));
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = poisson_log_lpmf(y[i] | eta[i]);
  }
}
