// Logit Model - using binomial-logit
//
// y ~ Bernoulli(p)
// p = X B
// b \sim cauchy(0, 2.5)
data {
  // number of observations
  int n;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[n];
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // beta prior
  real b_loc;
  real<lower = 0.0> b_scale;
}
parameters {
  // regression coefficient vector
  vector[k] b;
}
model {
  // priors
  b ~ cauchy(0.0, 2.5);
  // likelihood
  y ~ binomial_logit(1, X * b);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  // log-likelihood posterior
  vector[n] log_lik;
  // predicted probabilities
  vector[n] p;
  p = inv_logit(X * b);
  for (i in 1:n) {
    y_rep[i] = binomial_rng(1, p[i]);
    log_lik[i] = binomial_lpmf(y[i] | 1, p[i]);
  }
}
