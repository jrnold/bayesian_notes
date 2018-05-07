data {
  int N1;
  int n2;
  vector[N] y1;
  vector[N] y2;
  real<lower=0.> scale_mu;
  real<lower=0.> scale_delta;
  real<lower=1.> scale_sigma1;
  real<lower=1.> scale_sigma2;
}
model {
  mu ~ normal(0, scale_mu);

  sigma1 ~ exponential(loc_sigma1);
  sigma2 ~ exponential(loc_sigma2);
  y1 ~ student_t(mu, sigma1);
  y2 ~ student_t(mu + delta, sigma2);
}

