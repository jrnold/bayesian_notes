data {
  # number obs
  int n;
  vector[n] y;
  # number cols in X
  int k;
  # design matrix X
  matrix [n, k] X;
}
parameters {
  vector[k] b;
  real<lower = 0.0> sigma;
}
transformed parameters {
  # keep E(Y | X) for each obs
  vector[n] mu;
  mu = X * b;
}
model {
  # if no priors specified, then default flat priors
  # likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  # for each obs draw a y
  vector[n] y_rep;
  vector[n] log_lik;
  for (i in 1:n) {
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
