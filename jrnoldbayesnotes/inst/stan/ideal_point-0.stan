data {
  // number of items
  int N;
  // number of individuals
  int K;
  // observed votes
  int<lower = 0, upper = N * K> Y_obs;
  int y_idx_row[Y_obs];
  int y_idx_col[Y_obs];
  int y[Y_obs];
}
parameters {
  // item difficulties
  vector[N] alpha;
  // item discrimination
  vector[N] lambda;
  // unknown ideal points
  vector[K] theta;
}
transformed parameters {
  // create theta from observed and parameter ideal points
  vector[Y_obs] mu;
  for (i in 1:Y_obs) {
    int tmpN;
    int tmpK;
    tmpN = y_idx_row[i];
    tmpK = y_idx_col[i];
    mu[i] = alpha[tmpN] + lambda[tmpN] * theta[tmpK];
  }
}
model {
  lambda ~ normal(0., 2.5);
  theta ~ normal(0., 1.);
  y ~ binomial_logit(1, mu);
}
generated quantities {
  vector[Y_obs] log_lik;
  // int y_rep[Y_obs];
  for (i in 1:Y_obs) {
    log_lik[i] = binomial_logit_lpmf(y[i] | 1, mu[i]);
    // y_rep[i] = binomial_rng(1, inv_logit(mu[i]));
  }
}
