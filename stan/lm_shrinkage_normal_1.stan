/* Bayesian "Ridge" Regression

Linear regression with normal errors and normal prior on regression coefficients.

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
  // prior scale on global df
  real<lower=0.> df_tau;
  real<lower=0.> scale_tau;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector on standardized scale
  real alpha_z;
  vector[K] beta_z;
  // scale of regression errors
  real<lower=0.> sigma;
  // global scale
  real<lower=0.> tau;
}
transformed parameters {
  // expected value of the response
  vector[N] mu;
  // coefficients
  real alpha;
  vector[K] beta;

  alpha = scale_alpha * alpha_z;
  beta = tau * sigma * beta_z;
  mu = alpha + X * beta;
}
model {
  // hyperpriors
  tau ~ student_t(df_tau, 0., scale_tau);
  // priors
  sigma ~ exponential(rate_sigma);
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  // shrinkage parameter
  // no local scales so only one value
  real kappa;
  // numver of effective coefficients
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
    kappa = 1. / (1. + N * inv_sigma2 * tau2);
  }
  m_eff = K * (1 - kappa);
}
