# ALD
functions {
  real asymmetric_laplace_lpdf(real y, real mu, real sigma, real tau) {
    real lp;
    real z;
    lp = log(tau) + log(1 - tau) - log(sigma);
    z = (y - mu) / sigma;
    if (z < 0) {
      lp = lp - z * (tau - 1);
    } else {
      lp = lp - z * tau;
    }
    return lp
  }
}
