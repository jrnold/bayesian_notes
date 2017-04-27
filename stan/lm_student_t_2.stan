data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // beta prior
  real b_loc;
  real<lower = 0.0> b_scale;
  // sigma prior
  real sigma_scale;
}
parameters {
  // regression coefficient vector
  vector[k] b;
  // scale of the regression errors
  real<lower = 0.0> sigma;
  // 1 / \lambda^2
  vector<lower = 0.0>[n] inv_lambda2;
  real<lower = 1.0> nu;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  vector[n] mu;
  // tau are obs-level scale params
  vector[n] tau;
  mu = X * b;
  // need to use elmentwise division
  tau = sigma ./ sqrt(inv_lambda2);
}
model {
  real half_nu;
  half_nu = 0.5 * nu;
  // priors
  b ~ normal(b_loc, b_scale);
  nu ~ gamma(2, 0.1);
  sigma ~ cauchy(0, sigma_scale);
  inv_lambda2 ~ gamma(half_nu, half_nu);
  // likelihood
  y ~ normal(mu, tau);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  // log-likelihood values
  vector[n] log_lik;
  // use a single loop since both y_rep and log_lik are elementwise
  for (i in 1:n) {
    y_rep[i] = student_t_rng(nu, mu[i], sigma);
    log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
  }

}
