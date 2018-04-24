// Bernoulli-Logit GLM with binomial priors
data {
  // number of observations
  int<lower=0> N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
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
  // likelihood
  y ~ bernoulli_logit(eta);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = bernoulli_rng(inv_logit(eta[i]));
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta[i]);
  }
}
