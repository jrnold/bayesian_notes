data {
  int N;
  // response
  vector[N] y;
  // groups
  int J;
  int group[J];
  // covariates with random
  real<lower = 0.> Omega_regularization;
  real<lower = 0.> Omega_concentration;
  real<lower = 0.> Omega_shape;
  real<lower = 0.> Omega_scale;
  // intercept and coef scales
  real<lower = 0.> intercept_scale;
  real<lower = 0.> coef_scale;
  real<lower = 0.> sigma_scale;
  // this should include the intercept
  int K;
  vector[K] X[N];
}
parameters {
  cholesky_factor_corr[K] Omega_L;
  simplex[K] Omega_simplex;
  real<lower = 0.> Omega_var;
  vector[J] gamma;
  vector[K] b[J];
  real<lower = 0.> sigma;
}
transformed parameters {
  vector[N] mu;
  vector<lower = 0.>[K] Omega_s;
  cov_matrix[K] Omega;
  for (i in 1:N) {
    mu[i] = dot_product(b[group[i]], X[i]);
  }
  Omega_s = sqrt(Omega_simplex * Omega_var * K);
  Omega = diag_matrix(Omega_s) * Omega_L * Omega_L' * diag_matrix(Omega_s);
}
model {
  // covariance matrix priors
  Omega_L ~ lkj_corr_cholesky(Omega_regularization);
  Omega_simplex ~ dirichlet(rep_vector(Omega_concentration, K));
  Omega_var ~ gamma(Omega_shape, Omega_scale);
  // coef mean priors
  gamma[1] ~ normal(0., intercept_scale);
  for (k in 2:K) {
    gamma[k] ~ normal(0., coef_scale);
  }
  // coefficient
  b ~ multi_normal(gamma, Omega);
  // obs. variance
  sigma ~ cauchy(0., sigma_scale);
  // likelihood
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
