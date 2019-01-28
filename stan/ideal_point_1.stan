// ideal point model
// identification:
// - xi ~ hierarchical
// - except fixed senators
data {
  // number of individuals
  int N;
  // number of items
  int M;
  // observed votes
  int<lower = 0, upper = N * M> P;
  int y_idx_leg[P];
  int y_idx_vote[P];
  int y[P];
  // priors
  // on items
  real loc_alpha;
  real<lower = 0.> scale_alpha;
  real loc_beta;
  real<lower = 0.> scale_beta;
  // on legislators
  int N_xi_obs;
  int idx_xi_obs[N_xi_obs];
  vector[N_xi_obs] xi_obs;
  int N_xi_param;
  int idx_xi_param[N_xi_param];
  // prior on ideal points
  real loc_xi;
  real<lower = 0.> scale_xi;
}
parameters {
  // item difficulties
  vector[M] alpha;
  // item discrimination
  vector[M] beta;
  // unknown ideal points
  vector[N_xi_param] xi_param;
  // hyperpriors
  real<lower = 0.> tau;
  real<lower = 0.> zeta;
}
transformed parameters {
  vector[P] eta;
  vector[N] xi;
  xi[idx_xi_param] = xi_param;
  xi[idx_xi_obs] = xi_obs;
  for (i in 1:P) {
    eta[i] = alpha[y_idx_vote[i]] + beta[y_idx_vote[i]] * xi[y_idx_leg[i]];
  }
}
model {
  alpha ~ normal(loc_alpha, scale_alpha);
  beta ~ normal(loc_beta, scale_beta);
  xi_param ~ normal(loc_xi, scale_xi);
  y ~ bernoulli_logit(eta);
}
generated quantities {
  vector[P] log_lik;
  for (i in 1:P) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta[i]);
  }
}
