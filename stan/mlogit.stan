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
  prior_scale_a = rep_vector(10., M);
  for (m in 1:M) {
    prior_scale_b[m] = rep_vector(2.5, M);
  }
}
parameters {
  vector[M] a;
  matrix[M, K] b;
}
transformed parameters {
  vector[M] eta[N];
  for (n in 1:N) {
    eta[n] = a + b * X[n];
  }
}
model {
  a ~ normal(0, prior_scale_a);
  for (m in 1:M) {
    b[m] ~ normal(0, prior_scale_b[m]);
  }
  for (n in 1:N) {
    y[n] ~ categorical_logit(eta[n]);
  }
}
generated quantities {
  vector[N] log_lik;
  int ;
}
