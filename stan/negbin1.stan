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
  real b0;
  vector[K] b;
  real<lower = 0.0> phi;
}
transformed parameters {
  vector<lower = 0.0, upper = 1.0>[N] mu;
  mu = exp(b0 + X * b);
}
model {
  // priors
  b0 ~ cauchy(0.0, 10.0);
  b ~ cauchy(0.0, 2.5);
  phi ~ cauchy(0.0, 1.0);
  // likelihood
  y ~ neg_binomial_2(mu, phi);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpdf(y[i] | mu[i], phi);
  }
}
