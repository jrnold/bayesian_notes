data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // // beta prior
  // real b_loc;
  // real<lower = 0.0> b_scale;
  // // sigma prior
  // real sigma_scale;
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
  // b ~ normal(b_loc, b_scale);
  // sigma ~ cauchy(0, sigma_scale);
  // likelihood
  y ~ normal(mu, sigma);
  // the ~ is a shortcut
  // target += normal_lpdf(y | mu, sigma);
  // for (i in 1:n) {
  //   y[i] ~ normal(mu[i], sigma)
  // }
}
generated quantities {
  // // simulate data from the posterior
  // vector[n] y_rep;
  // // log-likelihood posterior
  // vector[n] log_lik;
  // for (i in 1:n) {
  //   y_rep[i] = normal_rng(mu[i], sigma);
  //   log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  // }
}
