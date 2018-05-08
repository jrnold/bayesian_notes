// Logit Model
//
// y ~ Bernoulli(p)
// p = a + X B
//
// Adjust for the
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
  vector<lower = 0.0>[K] x_scale;
}
transformed data {
  # default scales same as rstanarm
  # assume data is centered and scaled
  real<lower = 0.0> a_scale;
  real<lower = 0.0> b_scale_raw;
  vector<lower = 0.0>[K] b_scale;
  a_scale = 10.0;
  b_scale_raw = 2.5;
  b_scale = b_scale_raw * x_scale;
}
parameters {
  // regression coefficient vector
  real a_raw;
  vector[K] b_raw;
}
transformed parameters {
  real a;
  vector[K] b;
  vector<lower = 0.0, upper = 1.0>[N] p;
  b = b_raw * b_scale;
  p = inv_logit(a + X * b);
}
model {
  // priors
  a ~ normal(0.0, a_scale);
  b ~ normal(0.0, b_scale);
  // likelihood
  y ~ binomial(1, p);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = binomial_rng(1, p[i]);
    log_lik[i] = binomial_lpmf(y[i] | 1, p[i]);
  }
}
