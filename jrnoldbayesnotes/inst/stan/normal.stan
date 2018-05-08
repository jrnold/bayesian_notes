data {
  # number obs
  int n;
  vector[n] y;
  # priors
  real mu_loc;
  real<lower = 0.0> mu_scale;
  real sigma_scale;
}
parameters {
  real mu;
  real<lower = 0.0> sigma;
}
model {
  # priors on the parameters
  mu ~ normal(mu_loc, mu_scale);
  sigma ~ cauchy(0.0, sigma_scale);
  # likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  # for each obs draw a y
  vector[n] y_rep;
  for (i in 1:n) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}
