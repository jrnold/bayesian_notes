// Linear Model with Student-t Errors
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  vector<lower=0.>[K] scale_beta;
  real<lower=0.> loc_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  // regression scale
  real<lower=0.> omega;
  // 1 / lambda_i^2
  vector<lower = 0.0>[N] inv_lambda2;
  // degrees of freedom;
  // limit df = 2 so that there is a finite variance
  real<lower=2.> nu;
}
transformed parameters {
  vector[N] mu;

  mu = alpha + X * beta;
}
model {
  real half_nu;
  vector[N] sigma;

  // priors
  alpha ~ normal(0.0, scale_alpha);
  beta ~ normal(0.0, scale_beta);
  sigma ~ exponential(loc_sigma);
  nu ~ gamma(2, 0.1);
  half_nu = 0.5 * nu;
  inv_lambda2 ~ gamma(half_nu, half_nu);
  // observation variances
  for (n in 1:N) {
    sigma[n] = omega / sqrt(inv_lambda2[n]);
  }
  // likelihood with obs specific scales
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (n in 1:num_elements(y_rep)) {
    y_rep[n] = student_t_rng(nu, mu[n], omega);
  }
  for (n in 1:num_elements(log_lik)) {
    log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], omega);
  }
}
