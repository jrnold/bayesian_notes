// Logit Model
//
// y ~ Bernoulli(p)
// p = a + X B
// no priors
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
  matrix [N, K] X;
}
parameters {
  // regression coefficient vector
  real b0;
  vector[K] b;
}
transformed parameters {
  vector<lower = 0.0, upper = 1.0>[N] p;
  p = inv_logit(b0 + X * b);
}
model {
  // likelihood
  y ~ binomial(1, p);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] loglik;
  for (i in 1:N) {
    y_rep[i] = binomial_rng(1, p[i]);
    loglik[i] = binomial_lpmf(y[i] | 1, p[i]);
  }
}
