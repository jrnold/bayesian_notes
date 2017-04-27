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
  vector[n] mu;
  // tau are obs-level scale params
  mu = X * b;
}
model {
  // priors
  b ~ normal(b_loc, b_scale);
  sigma ~ cauchy(0, sigma_scale);
  // likelihood
  y ~ double_exponential(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  // log-likelihood values
  vector[n] log_lik;
  // use a single loop since both y_rep and log_lik are elementwise
  for (i in 1:n) {
    y_rep[i] = double_exponential_rng(mu[i], sigma);
    log_lik[i] = double_exponential_lpdf(y[i] | mu[i], sigma);
  }

}
