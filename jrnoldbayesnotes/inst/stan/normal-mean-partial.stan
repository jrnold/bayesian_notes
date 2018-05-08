data {
  int N;
  // response
  vector[N] y;
  // groups
  int J;
  int group[J];
  // priors
  // scale on tau
  real<lower = 0.> tau_scale;
  // prior on loc and scale
  real gamma_loc;
  real<lower = 0.> gamma_scale;
  // prior on obs variance
  real<lower = 0.> sigma_scale;
}
parameters {
  real<lower = 0.> tau;
  real gamma;
  vector[J] mu;
  real<lower = 0.> sigma;
}
model {
  tau ~ cauchy(0., tau_scale);
  gamma ~ normal(gamma_loc, gamma_scale);
  mu ~ normal(gamma, tau);
  sigma ~ cauchy(0., sigma_scale);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}
