data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // beta prior
  real b_loc;
  real<lower = 0.0> b_scale;
  // sigma prior
  real sigma_scale;
}
parameters {
  // regression coefficient vector
  vector[k] b;
  // scale of the regression errors
  real<lower = 0.0> sigma;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[n] mu;
  mu = X * b;
}
model {
  // priors
  b ~ normal(0.0, tau);
  tau ~ normlal(0.0, )
  sigma ~ cauchy(0, sigma_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  // log-likelihood posterior
  vector[n] log_lik;
  for (i in 1:n) {
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
