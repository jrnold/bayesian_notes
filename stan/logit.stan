// Logit Model
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
}
parameters {
  // regression coefficient vector
  vector[k] b;
}
transformed parameters {
  vector<lower = 0.0, upper = 1.0>[n] p;
  p = inv_logit(X * b);
}
model {
  // priors
  b ~ cauchy(0, 2.5);
  // likelihood
  y ~ binomial(1, p);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  // log-likelihood posterior
  vector[n] loglik;
  for (i in 1:n) {
    y_rep[i] = binomial_rng(1, p[i]);
    loglik[i] = binomial_lpmf(y[i] | 1, p[i]);
  }
}
