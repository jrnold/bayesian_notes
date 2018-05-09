/*
Zero-Inflated Negative Binomial Regression
*/
functions {
  /* Zero Inflated Negative Binomial Probability Mass Function

  The log zero-inflated negative binomial probability mass of `y` given zero-inflation weight `theta`,
  log-location `eta` and inverse overdispersion control `phi`.

  @param y Response
  @param theta Zero-inflation weight
  @param eta Log mean parameter
  @param phi Inverse overdispersion control parameter
  @return Log probability mass

  */
  real zinfl_neg_binomial_2_log_lpmf(int y, real theta, real eta, real phi) {
    if (y == 0) {
      return log_sum_exp(bernoulli_lpmf(1 | theta),
                            bernoulli_lpmf(0 | theta) +
                            neg_binomial_2_log_lpmf(y | eta, phi));
    } else {
      return bernoulli_lpmf(0 | theta) +
        neg_binomial_2_log_lpmf(y | eta, phi);
    }
  }
  /* Zero Inflated Negative Binomial Random Number

  Generate a zero-inflated negative binomial variate
  with zero-inflation weight `theta`,
  log-location `eta`, and inverse overdispersion
  control `phi`;
  may only be used in generated quantities block.
  `eta` must be less than 29 log 2.

  @param theta Zero-inflation weight
  @param mu Mean parameter
  @param phi Inverse overdispersion control parameter
  @return Random variate from the zero-inflated negative binomial distribution

  */
  int zinfl_neg_binomial_2_rng(real theta, real mu, real phi) {
    int z;
    z = bernoulli_rng(theta);
    if (z) {
      return neg_binomial_2_rng(mu, phi);
    } else {
      return z;
    }
  }
}
data {
  // number of observations
  int<lower=0> N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0> y[N];
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  real<lower=0.> scale_beta;
  real<lower=0.> shape1_theta;
  real<lower=0.> shape2_theta;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  // 1 / sqrt(phi)
  real<lower=0.> inv_sqrt_phi;
  // zero-inflated mixture
  real<lower=0.,upper=1.> theta;
}
transformed parameters {
  vector[N] eta;
  real<lower=0.> phi;

  phi = 1 / inv_sqrt_phi ^ 2;
  eta = alpha + X * beta;
}
model {
  // priors
  // see https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
  inv_sqrt_phi ~ normal(0., 1.);
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  theta ~ beta(shape1_theta, shape2_theta);
  for (n in 1:N) {
    y[n] ~ zinfl_neg_binomial_2_log_lpmf(theta, eta[n], phi);
  }
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = zinfl_neg_binomial_2_rng(theta, exp(eta[i]), phi);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = zinfl_neg_binomial_2_log_lpmf(y[i] | theta, eta[i], phi);
  }
}
