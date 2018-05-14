// Faster Ridge regression
// Directly sample the posterior distribution of beta using the closed-form solution
data {
  // number of observations
  int<lower=0> N;
  int<lower=0> K;
  // Precomputed X'X, X'y
  matrix[K, K] XX;
  matrix[K, K] XX_inv;
  vector[K] Xy;
  // priors on alpha
  real<lower=0.> loc_sigma;
  real<lower=0.> df_lambda;
}
parameters {
  // regression coefficient vector
  vector[K] beta;
  real<lower=0.> lambda;
  real<lower=0.> sigma;
}
transformed parameters {
  vector[K] beta_mean;
  cov_matrix[K] beta_cov;
  {
    matrix[K, K] M = inverse(XX - diag_matrix(rep_vector(lambda, K)));
    beta_mean = M * Xy;
    beta_cov = sigma ^ 2 * quad_form_sym(XX_inv, M);
  }
}
model {
  beta ~ multi_normal(beta_mean, beta_cov);
  sigma ~ exponential(loc_sigma);
  lambda ~ chi_square(df_lambda);
}
generated quantities {
}
