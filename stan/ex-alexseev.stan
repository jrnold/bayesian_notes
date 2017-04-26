data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // marfx
  // indexes of main and interaction coef
  int idx_b_slavicshare;
  int idx_b_slavicshare_changenonslav;
  int M;
  vector[M] changenonslav;
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
  b ~ normal(b_loc, b_scale);
  sigma ~ cauchy(0, sigma_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  # hardcoded marginal effectx
  vector[M] dydx;
  dydx = b[idx_b_slavicshare] + b[idx_b_slavicshare_changenonslav] * changenonslav;
}
