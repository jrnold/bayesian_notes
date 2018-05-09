data {
  // number of rats
  int n;
  // number of observations
  int m;
  // size of rat
  vector[m] y;
  // times for each rat
  vector<lower=0,upper=n>[m] x;
  // rat ID
  int<lower=1,upper=m> id[m];
  // priors
  real m_alpha;
  real m_beta;
  real<lower=0.> s_alpha;
  real<lower=0.> s_beta;
}
parameters {
  vector[n] alpha;
  vector[n] beta;
  real<lower=0.> sigma;
}
model {
  vector[m] beta_vec;
  vector[m] alpha_vec;

  // create alpha and beta vectors the same size as y
  // this is more efficient than calculating the normal distribution n times.
  for (i in 1:m) {
    beta_vec[i] = beta[id[i]];
    alpha_vec[i] = alpha[id[i]];
  }
  y ~ normal(alpha_vec + beta_vec * x, sigma);
  beta ~ normal(m_beta, s_beta);
  alpha ~ normal(m_alpha, s_alpha);
}
