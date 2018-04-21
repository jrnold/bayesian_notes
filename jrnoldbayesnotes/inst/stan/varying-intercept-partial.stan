data {
  int N;
  // response
  vector[N] y;
  // groups
  int J;
  int group[J];
  // should include the intercept
  int K;
  matrix[N, K] X;
  // covariates with random
  real<lower = 0.> tau_scale;
  // intercept and coef scales
  real intercept_loc;
  real<lower = 0.> intercept_scale;
  vector[K] coef_loc;
  vector<lower = 0.>[K] coef_scale;
  real<lower = 0.> sigma_scale;
}
parameters {
  real<lower = 0.> tau;
  real gamma;
  vector[J] a;
  vector[K] b;
  real<lower = 0.> sigma;
}
transformed parameters {
  vector[N] mu;
  // Calculate mu for each
  {
    vector[N] tmp;
    tmp = X * b;
    for (i in 1:N) {
      mu[i] = a[group[i]] + tmp[i];
    }
  }
}
model {
  tau ~ cauchy(0., tau_scale);
  gamma ~ normal(intercept_loc, intercept_scale);
  a ~ normal(gamma, tau);
  b ~ normal(coef_loc, coef_scale);
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
