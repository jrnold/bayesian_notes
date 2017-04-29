// Logit Model
//
// y ~ poisson(lambda)
// lambda = exp(b0 + X b)
// b0 \sim cauchy(0, 10)
// b \sim cauchy(0, 2.5)
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
  vector<lower = 0.0>[N] lambda;
  lambda = exp(b0 + X * b);
}
model {
  // priors
  b0 ~ cauchy(0.0, 10.0);
  b ~ cauchy(0.0, 2.5);
  // likelihood
  y ~ poisson(lambda);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = poisson_rng(lambda[i]);
    log_lik[i] = poisson_lpdf(y[i] | lambda[i]);
  }
}
