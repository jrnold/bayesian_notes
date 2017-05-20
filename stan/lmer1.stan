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
  // should include the intercept
  int K;
  vector[K] X[N];
}
parameters {
  cholesky_factor_corr[K] Omega_L;
  simplex[K] Omega_simplex;
  real<lower = 0.> Omega_s;
  vector[J] gamma;
  vector[K] b[J];
  real<lower = 0.> sigma;

}
transformed parameters {
  vector[N] mu;
  cov_matrix[K] Omega;
  for (i in 1:N) {
    mu[i] = dot_product(b[group[i]], X[i]);
  }
  Omega = diag_pre_multiply(K * pow(Omega_s, -2) * Omega_simplex, Omega_L * Omega_L');
}
model {
  // covariance matrix
  Omega_L ~ lkj_corr_cholesky(Omega_regularization);
  Omega_simplex ~ dirichlet(rep_vector(Omega_concentration, K));
  Omega_s ~ gamma(Omega_shape, Omega_scale);
  gamma[1] ~ normal(0., intercept_scale);
  for (k in 2:K) {
    gamma[k] ~ normal(0., coef_scale);
  }
  // todo
  sigma ~ cauchy(0., sigma_scale);
  b ~ multi_normal(gamma, Omega);
  y ~ normal(mu, sigma);
}
