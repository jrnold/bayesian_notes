// Rare-Events Logit Model with Prior Correction
data {
  // number of observations
  int N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];
  // need to pass mean of y since it is hard to cast integer to real types
  // in stan
  real y_mean;
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
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
transformed data {
  real correction;

  // log((y_mean) / (1 - ymean) * tau / (1 - tau))
  correction = log(y_mean) - log1m(y_mean) + log1m(tau) - log(tau);
}
parameters {
  // regression coefficient vector
  real alpha_raw;
  vector[K] beta;
}
transformed parameters {
  // sampling corrected intercept
  real alpha;
  // logit-scale means
  vector[N] eta;

  alpha = alpha_raw - correction;
  eta = alpha + X * beta;
}
model {
  // priors
  alpha_raw ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  // likelihood
  y ~ bernoulli_logit(eta);
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
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = bernoulli_lpmf(y[i] | inv_logit(eta[i]));
  }
}
