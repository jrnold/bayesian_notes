/*
Zero-Inflated Negative Binomial Regression
*/
functions {
  /*

  The zero-inflated Poisson binomial log probability mass of `y` given zero-inflation weight `theta`,
  log-location `eta` and inverse overdispersion control `phi`.

  @param y Response
  @param theta Zero-inflation weight
  @param eta Log mean parameter
  @return Log probability mass

  */
  real zinfl_poisson_log_lpmf(int y, real theta, real eta) {
    if (y == 0) {
      return log_sum_exp(bernoulli_lpmf(1 | theta),
                            bernoulli_lpmf(0 | theta) +
                            poisson_log_lpmf(y | eta));
    } else {
      return bernoulli_lpmf(0 | theta) +
        poisson_log_lpmf(y | eta);
    }
  }
  /*

  Generate a zero-inflated Poisson variate
  with zero-inflation weight `theta`, and log-location `eta`.

  @param theta Zero-inflation weight
  @param eta Log mean parameter
  @return Random variate from the zero-inflated negative binomial distribution

  */
  int zinfl_poisson_rng(real theta, real lambda) {
    int z;
    z = bernoulli_rng(theta);
    if (z) {
      return poisson_rng(lambda);
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
    y[n] ~ zinfl_poisson_log(theta, eta[n]);
  }
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = zinfl_poisson_rng(theta, exp(eta[i]));
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = zinfl_poisson_log_lpmf(y[i] | theta, eta[i]);
  }
}
