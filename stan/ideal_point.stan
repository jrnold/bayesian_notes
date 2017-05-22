data {
  // number of items
  int N;
  // number of individuals
  int K;
  // observed votes
  int<lower = 0, upper = N> N_obs;
  int y_idx_row[N_obs];
  int y_idx_col[N_obs];
  int y[N_obs];
  // ideal points
  // for identification, some ideal points are fixed
  int<lower = 0, upper = K> K_obs;
  int<lower = 0, upper = K> K_param;
  int<lower = 1, upper = K> theta_obs_idx[K_obs];
  int theta_obs[K_obs];
  int<lower = 1, upper = K> theta_param_idx[K_param];
}
parameters {
  // item difficulties
  vector[N] alpha;
  // item discrimination
  vector[N] lambda;
  // unknown ideal points
  vector[K_param] theta_param;
}
transformed parameters {
  // create theta from observed and parameter ideal points
  vector[K] theta;
  vector[N_obs] mu;
  for (k in 1:K_param) {
    theta[theta_param_idx[k]] = theta_param[k];
  }
  for (k in 1:K_obs) {
    theta[theta_obs_idx[k]] = theta_obs[k];
  }
  for (i in 1:N_obs) {
    int tmpN;
    int tmpK;
    tmpN = y_idx_row[i];
    tmpK = y_idx_col[i];
    mu[i] = alpha[tmpN] + lambda[tmpN] * theta[tmpK];
  }
}
model {
  lambda ~ normal(0., 2.5);
  theta_param ~ normal(0., 1.);
  y ~ binomial_logit(1, mu);
}
generated quantities {
  vector[N_obs] log_lik;
  // int y_rep[N_obs];
  for (i in 1:N_obs) {
    log_lik[i] = binomial_logit_lpmf(y[i] | 1, mu[i]);
    // y_rep[i] = binomial_rng(1, inv_logit(mu[i]));
  }
}
