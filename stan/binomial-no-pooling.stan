/* Binomial Model (No pooling)

  A binomial model for $i = 1, \dots, N$, no pooling:
  $$
  p(y_i | n_i, \mu_i) &\sim \mathsf{Binomial}(y_i | n_i, \mu_i) \\
  \mu_i &= \logit^{-1}(\eta_i) \\
  p(\eta_i) &\sim \mathsf{Normal}^+(0, 10)
  $$

*/
data {
  int N;
  int y[N];
  int k[N];
  // new data
  int y_new[N];
  int k_new[N];
}
parameters {
  vector[N] eta;
}
model {
  eta ~ normal(0., 10.);
  //  y ~ binomial(inv_logit(eta));
  // binomial_logit
  y ~ binomial_logit(k, eta);
}
generated quantities {
  int y_rep[N];
  vector[N] log_lik;
  vector[N] log_lik_new;
  vector<lower = 0., upper = 1.>[N] mu;
  mu = inv_logit(eta);
  for (n in 1:N) {
    y_rep[n] = binomial_rng(k[n], mu[n]);
    log_lik[n] = binomial_logit_lpmf(y[n] | k[n], eta[n]);
    log_lik_new[n] = binomial_logit_lpmf(y_new[n] | k_new[n], eta[n]);
  }
}
