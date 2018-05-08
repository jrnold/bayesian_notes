data {
  int N;
  // response
  vector[N] y;
  // groups
  int J;
  int group[J];
  // priors
  real<lower = 0.> sigma_scale;
}
parameters {
  vector[J] mu;
  real<lower = 0.> sigma;
}
model {
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
