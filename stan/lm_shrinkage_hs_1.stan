// Linear Model with Normal Errors
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
  // prior scale on global
  real<lower=0.> loc_tau;
  // prior on regression error distribution
  real<lower=0.> loc_sigma;
  // location of prior on lambda
  real<lower=0.> loc_lambda;
  // degrees of freedom of hyperprior of lambda.
  // If d = 1, then lambda_i ~ Cauchy() and it is horseshoe prior
  real<lower=0.> d;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  real<lower=0.> sigma;
  // hyper-parameters of coefficients
  real<lower=0.> tau;
  vector<lower=0.>[K] lambda;
}
transformed parameters {
  vector[N] mu;

  mu = alpha + X * beta;
}
model {
  // hyperpriors
  lambda ~ student_t(d, 0, loc_lambda);
  tau ~ exponential(loc_tau);
  // priors
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., tau * lambda);
  sigma ~ exponential(loc_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
