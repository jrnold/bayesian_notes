data {
  // number of observations
  int N;
  // response vector
  vector[N] y;
  // number of columns in the design matrix X
  int K;
  // design matrix X
  matrix [N, K] X;
}
transformed data {
  real<lower = 0.> y_sd;
  real a_pr_scale;
  real sigma_pr_scale;
  real tau_pr_scale;
  y_sd = sd(y);
  sigma_pr_scale = y_sd * 5.;
  a_pr_scale = 10.;
}
parameters {
  // regression coefficient vector
  real a;
  vector[K] b;
  // scale of the regression errors
  real<lower = 0.> sigma;
  // scale of normal prior on coefficients
  real<lower = 0.> tau;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[N] mu;
  mu = a + X * b;
}
model {
  // priors
  a ~ normal(0., a_pr_scale);
  b ~ normal(0., tau);
  tau ~ cauchy(0., 1.);
  sigma ~ cauchy(0., sigma_pr_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
