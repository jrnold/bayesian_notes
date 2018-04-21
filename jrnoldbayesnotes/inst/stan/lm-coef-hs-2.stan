data {
  // number of observations
  int N;
  // response vector
  vector[N] y;
  // number of columns in the design matrix X
  int K;
  // design matrix X
  matrix [N, K] X;
  // degress of freedom for local and global scales
  real<lower = 0.> df_local;
  real<lower = 0.> df_global;
}
transformed data {
  real<lower = 0.> y_sd;
  real a_pr_scale;
  real sigma_pr_scale;
  y_sd = sd(y);
  sigma_pr_scale = y_sd * 5.;
  a_pr_scale = 10.;
}
parameters {
  // regression coefficient vector
  real a;
  vector[K] b_raw;
  // scale of the regression errors
  real<lower = 0.> sigma;
  // global scale for coefficients
  real<lower = 0.> tau;
  // local scales of coefficients
  vector<lower = 0.>[K] lambda;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[N] mu;
  vector[K] b;
  b = b_raw * tau .* lambda;
  mu = a + X * b;
}
model {
  // priors
  lambda ~ student_t(df_local, 0., 1.);
  a ~ normal(0., a_pr_scale);
  b_raw ~ normal(0., 1.);
  tau ~ student_t(df_global, 0., 1.);
  sigma ~ cauchy(0., sigma_pr_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
