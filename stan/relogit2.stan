// Rare-Events Logit Model with Prior Correction
data {
  // number of observations
  int N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];

  // number of columns in the design matrix X
  int K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  real<lower=0.> scale_beta;
  // rare-event logit correction
  // population proportion of outcomes
  real<lower=0., upper=1.> tau;
  // need to pass mean of y since it is hard to cast integer to real types
  // in stan
  real y_mean;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
transformed data {
  // weights
  vector[N] w;
  for (n in 1:N) {
    if (y[n]) {
      w[n] = tau / y_mean;
    } else {
      w[n] = (1. - tau) / (1. - y_mean);
    }
  }
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
}
transformed parameters {
  // logit-scale means
  vector[N] eta;

  eta = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0.0, scale_alpha);
  beta ~ normal(0.0, scale_beta);
  // manually calculate the weighted log-likelihood
  for (n in 1:N) {
    target += w[n] * bernoulli_logit_lpmf(y[n] | eta[n]);
  }
}
generated quantities {
  // simulate data from the posterior
  // actually simulate proportions rather than the outcomes - these are easy
  // enough to create.
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;

  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = bernoulli_rng(inv_logit(eta[i]));
  }
  // we don't need to weight the individual log-likelihoods
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = bernoulli_lpmf(y[i] | inv_logit(eta[i]));
  }
}
