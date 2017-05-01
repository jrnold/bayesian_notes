# ALD
functions {
  real asymmetric_laplace_lpdf(real y, real mu, real sigma, real tau) {
    real lp;
    real z;
    lp = log(tau) + log(1 - tau) - log(sigma);
    z = (y - mu) / sigma;
    if (z <= 0) {
      lp = lp - z * (tau - 1);
    } else {
      lp = lp - z * tau;
    }
    return lp
  }
  real asymmetric_laplace_cdf(real y, real mu, real sigma, real tau) {
    real p;
    z = (y - mu) / sigma;
    if (z <= 0) {
      p = log(tau) * (1 - tau) * z;
    } else {
      p = 1.0 (1.0 - tau) * exp(- tau * z);
    }
    return p;
  }
  # see https://en.wikipedia.org/wiki/Generalized_normal_distribution
  # version 1
  real generalized_normal_lpdf(real y, real mu, real alpha, real beta) {
    log(beta) - log(2) - log(alpha) - lgamma(1. / beta) - beta * (fabs(y - mu) / abs)
  }
  real generalized_normal_cdf(real y, real mu, real alpha, real beta)   {
    real sgn;
    real out;
    out = 0.5;
    if (x > mu) {
      sgn = 1;
    } else if (x < mu) {
      sgn = -1;
    } else {
      return out;
    }
    out = (out + 0.5 * sgn * lgamma_p(1. / beta, (fabs(x - mu) / alpha) ^ beta)


  }
}
