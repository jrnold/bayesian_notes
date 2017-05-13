// Negative Binomial Model
//
// y ~ negbin(mu, phi)
// mu = exp(b0 + X b)
// b0 \sim cauchy(0, 10)
// b \sim cauchy(0, 2.5)
data {
  // number of observations
  int N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0> y[N];
  // number of columns in the design matrix X
  int K;
  // design matrix X
  matrix [N, K] X;
}
parameters {
  // regression coefficient vector
  real a;
  vector[K] b;
  real<lower = 0.> reciprocal_phi;
}
transformed parameters {
  vector[N] eta;
  real<lower = 0.> phi;
  phi = 1. / reciprocal_phi;
  eta = a + X * b;
}
model {
  // priors
  a ~ normal(0., 10.);
  b ~ normal(0., 2.5);
  reciprocal_phi ~ cauchy(0., 5.);
  // likelihood
  y ~ neg_binomial_2_log(eta, phi);
}
generated quantities {
  // expected value
  vector[N] mu;
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  mu = exp(eta);
  for (i in 1:N) {
    y_rep[i] = neg_binomial_2_log_rng(eta[i], phi);
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | eta[i], phi);
  }
}
