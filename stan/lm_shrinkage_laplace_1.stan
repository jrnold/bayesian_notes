/*
  Linear Model with Laplace Prior on Coefficients  (Bayesian Lasso)

*/
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
  // Half-Student-t prior on tau
  real<lower=0.> df_tau;
  real<lower=0.> scale_tau;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression intercept
  real alpha_z;
  // regression coefficients
  vector[K] beta_z;
  // scale of regression errors
  real<lower=0.> sigma;
  // hyper-parameters of coefficients
  real<lower=0.> tau;
  vector<lower=0.>[K] lambda2;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;

  alpha = scale_alpha * alpha_z;
  beta = tau * sqrt(lambda2) .* beta_z;
  mu = alpha + X * beta;
}
model {
  // hyperpriors
  tau ~ student_t(df_tau, 0., scale_tau * sigma);
  lambda2 ~ exponential(0.5);
  // priors
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  sigma ~ exponential(rate_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  // shrinkage factors
  vector[K] kappa;
  // number of effective coefficients
  real m_eff;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
  {
    real inv_sigma2 = 1. / sigma ^ 2;
    real tau2 = tau ^ 2;
    kappa = 1. ./ (1. + N * inv_sigma2 * tau2 * lambda2);
  }
  m_eff = K - sum(kappa);
}
