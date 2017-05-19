data {
  int N;
  vector y[N];
  vector[N] y;
  vector[N] x;
}
parameter {
  real a;
  real b;
  real<lower = 0.> sigma;
}
model {
  a ~ normal(0, 10.);
  b ~ normal(0, 10.);
  y ~ normal(mu, sigma);
}
