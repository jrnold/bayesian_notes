// ideal point model
// identification:
data {
  // Total legislators
  int N;
  // number of items
  int M;
  // observed votes
  int<lower = 0, upper = N * M> P;
  int y_idx_leg[P];
  int y_idx_vote[P];
  int y[P];
  // priors on items
  real loc_alpha;
  real<lower = 0.> scale_alpha;
  real loc_beta;
  real<lower = 0.> scale_beta;
  // Number of legislators with fixed ideal points
  int<lower=0, upper=N> N_obs;
  int idx_xi_obs[N_obs];
  vector[N_obs] xi_obs;
  int idx_xi_param[N - N_obs];
  // prior on ideal point parameters
  real loc_xi;
  real<lower = 0.> scale_xi;
  real<lower = 1.> df_xi;
}
transformed data {
  int N_param;
  N_param = N - N_obs;
}
parameters {
  // item difficulties
  vector[M] alpha_z;
  // item discrimination
  vector[M] beta_z;
  // unknown ideal points
  vector[N_param] xi_param_z;
}
transformed parameters {
  // create xi from observed and parameter ideal points
  matrix[N, M] eta;
  // item difficulties
  vector[M] alpha;
  // item discrimination
  vector[M] beta;
  // this is extra mem, but convenient
  vector[N] xi;

  alpha = loc_alpha + scale_alpha * alpha_z;
  beta = loc_beta + scale_beta * beta_z;

  // fill in xi vector
  for (i in 1:N_obs) {
    xi[idx_xi_obs[i]] = xi_obs[i];
  }
  for (i in 1:N_param) {
    xi[idx_xi_param[i]] = loc_xi + scale_xi * xi_param_z[i];
  }

  for (i in 1:N) {
    for (j in 1:M) {
      eta[i, j] = alpha[j] + beta[j] * xi[i];
    }
  }
}
model {
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  xi_param_z ~ student_t(df_xi, 0., 1.);
  for (p in 1:P) {
    y[p] ~ bernoulli_logit(eta[y_idx_leg[p], y_idx_vote[p]]);
  }
}
generated quantities {
  vector[P] log_lik;
  for (p in 1:P) {
    log_lik[p] = bernoulli_logit_lpmf(y[p] | eta[y_idx_leg[p], y_idx_vote[p]]);
  }
}
