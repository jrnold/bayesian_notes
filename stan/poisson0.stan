// Poisson Model
//
// y ~ Poisson(p)
// lambda = b0 + b
// b0 \sim cauchy(0, 10)
// b \sim cauchy(0, 2.5)
data {
  // number of observations
  int N;
  // response
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
}
transformed parameters {
  vector<lower = 0.0, upper = 1.0>[N] lambda;
  lambda = exp(b0 + X * b);
}
model {
  // likelihood
  y ~ poisson(lambda);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = poisson_rng(1, p[i]);
    log_lik[i] = poisson_lpmf(y[i] | lambda[i]);
  }
}
