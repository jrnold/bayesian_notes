/* Binomial Model

  A binomial model for $i = 1, \dots, N$, with complete pooling
  $$
  \begin{aligned}[t]
  p(y_i | n_i, \mu) &\sim \mathsf{Binomial}(n_i, \mu) \\
  \mu &= \logit^{-1}(\eta) \\
  p(\eta) &\sim \mathsf{Normal}^+(0, 10)
  \end{aligned}
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
  real eta;
}
model {
  eta ~ normal(0., 10.);
  y ~ binomial_logit(k, eta);
}
generated quantities {
  int y_rep[N];
  vector[N] log_lik;
  vector[N] log_lik_new;
  real<lower = 0., upper = 1.> mu;
  mu = inv_logit(eta);
  for (n in 1:N) { //
    y_rep[n] = binomial_rng(k[n], mu);
    log_lik[n] = binomial_logit_lpmf(y[n] | k[n], eta);
    log_lik_new[n] = binomial_logit_lpmf(y_new[n] | k_new[n], eta);
  }
}
