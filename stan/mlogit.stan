data {
  int<lower = 0> N;
  // categories
  int<lower = 2> M;
  int<lower = 1, upper = M> y[N];
  // covariates
  int<lower = 0> K;
  vector[K] X[N];
}
transformed data {
  vector<lower = 0.>[M] prior_scale_a;
  vector<lower = 0.>[K] prior_scale_b[M];
  prior_scale_a = rep_vector(M, 10.);
  for (m in 1:M) {
    prior_scale_b[m] = rep_vector(K, 2.5);
  }
}
parameters {
  vector[M] a;
  vector[M, K] b;
}
transformed parameters {
  vector[M] eta[N];
  for (n in 1:N) {
    eta[n] = a + X[n] * b;
  }
}
model {
  a ~ normal(0, prior_scale_a)
  for (m in 1:M) {
    b[m] ~ normal(0, prior_scale_b[m]);
  }
  for (n in 1:N) {
    y[n] ~ ctegorical_logit(eta);
  }
}
generated quantities {
  // log_lik
  // yrep
}
