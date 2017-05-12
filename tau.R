#' $$
#' \begin{aligned}[t]
#' E(\kappa_j | \tau, \sigma) &= \frac{1}{1 + \sigma^{-1} \tau \sqrt{n}} , \\
#' Var(\kappa_j | \tau, \sigma) &= \frac{\sigma^{-1} \tau \sqrt{n}}{2(1 + \sigma^{-1} \tau \sqrt{n})^2} \\
#' \end{aligned}
#' $$
kappa_moments <- function(tau, sigma, n) {
  stn <- sigma ^ -1 * tau * sqrt(n)
  list(mean = 1 / (1 + stn),
       var = stn / (2 * (1 + stn) ^ 2))
}

#' $$
#' \begin{aligned}[t]
#' E(m_{eff} | \tau, \sigma) &= \frac{1}{1 + \sigma^{-1} \tau \sqrt{n}} D, \\
#' Var(m_{eff} | \tau, \sigma) &= \frac{\sigma^{-1} \tau \sqrt{n}}{2(1 + \sigma^{-1} \tau \sqrt{n})^2} D . \\
#' \end{aligned}
#' $$
meff_moments <- function(tau, sigma, n, D = 1) {
  stn <- sigma ^ -1 * tau * sqrt(n)
  list(mean = 1 / (1 + stn) * D,
       var = stn / (2 * (1 + stn) ^ 2) * D)
}

#' $$
#' tau_0 = \frac{p_0}{D - p_0} \frac{\sigma}{\sqrt{n}}
#' $$
#' Van der plas suggest the following
#' $$
#' \tau_0 = \frac{p_0}{D},
#' $$
#' or,
#' $$
#' \tau_0 = \frac{p_0}{D} \sqrt{log(D / p_0)} .
#' $$
#'
tau_prior <- function(p, D = 1, sigma = 1, n = D, method = 1) {
  switch(method,
         p / (D - p) * sigma / sqrt(n),
         p / D,
         p / D * sqrt(log( n / p)),
         1,
         sigma)
}

# Simulate prior number of effective priors
sim_meff <- function(fun_lambda, fun_tau, sigma = 1, D = 1, n = 1,
                     sims = 1000) {
  precision <- sigma ^ -2
  f <- function(tau) {
    lambda <- fun_lambda(D)
    kappa <- 1 / (1 + n * precision * tau ^ 2 * lambda ^ 2)
    sum(1 - kappa)
  }
  map_dbl(fun_tau(sims), f)
}

# Kappa
# kappa_j = \frac{1}{1 + n \sigma^{-2} \tau^2 \lambda_j^2}
# inverse
# lambda_j(kappa_j) = \frac{1}{(1 + n \sigma^{-2} \tau^2) \kappa_j}
kappa_funs <- function(tau = 1, sigma = 1, n = 1) {
  precision <- sigma ^ -2
  list(
    # kappa(lambda)
    fun = function(lambda) {
      1 / (1 + n * precision * tau ^ 2 * lambda ^ 2)
    },
    # lambda(kappa)
    inv_scale = function(x) {
      sqrt(1 / ((1 + n * precision * tau ^ 2) * kappa))
    },
    # d(lambda(kappa)) / d kappa
    jacobian_scale = function(kappa) {
      (1 + n * precision * tau ^ 2) ^ (-0.5) * kappa ^ (-1.5)
    },
    # lambda^2(kappa)
    inv_var = function(kappa) {
      1 / ((1 + n * precision * tau ^ 2) * kappa)
    },
    # d(lambda^2(kappa)) / d kappa
    jacobian_var = function(kappa) {
      (1 + n * precision * tau ^ 2) ^ (-1) * kappa ^ (-2)
    },
    # lambda^{-2}(kappa)
    inv_prec = function(kappa) {
      (1 + n * precision * tau ^ 2) * kappa
    },
    # d(lambda^{-2}(kappa)) / d kappa
    jacobian_prec = function(kappa) {
      (1 + n * precision * tau ^ 2)
    }
  )
}

# Distribution over Kappa from distribution over lambda
kappa_dist <- function(dlambda, tau = 1, sigma = 1, n = 1,
                       type = c("scale", "variance", "precision"), ...) {
  type <- match.arg(type)
  fkappa <- kappa_funs(tau = tau, sigma = sigma, n = n)
  if (type == "scale") {
    finv <- fkappa$inv_scale
    jacobian <- fkappa$jacobian_scale
  } else if (type == "variance") {
    finv <- fkappa$inv_var
    jacobian <- fkappa$jacobian_var
  } else {
    finv <- fkappa$inv_prec
    jacobian <- fkappa$jacobian_prec
  }
  function(x) {
    dlambda(finv(x), ...) * jacobian(x)
  }
}


# Simulate prior m_eff
sim_meff <- function(fun_lambda, fun_tau = function(n) rep(1, n),
                     sigma = 1, D = 1, n = 1, sims = 1000) {
  precision <- 1 / sigma ^ 2
  f <- function(tau) {
    lambda <- fun_lambda(D)
    kappa <- 1 / (1 + n * precision * tau ^ 2 * lambda ^ 2)
    sum(1 - kappa)
  }
  map_dbl(fun_tau(sims), f)
}

# simulate prior kappa
sim_kappa <- function(fun_lambda, fun_tau = function(n) rep(1, n), sigma = 1,
                     n = 1, sims = 1000) {
  precision <- 1 / sigma ^ 2
  lambda <- fun_lambda(sims)
  tau <- fun_tau(sims)
  1 / (1 + n * precision * tau ^ 2 * lambda ^ 2)
}

# Find the shrinkage parameters of
# beta_j = (1 - \kappa_j) \hat\beta_j^{(MLE)}
beta_shrunk <- function(beta, X, y) {
  beta_hat <- lm.fit(X, y)$coefficients
  # trun to [0, 1]
  pmax(pmin(1, (beta_hat - beta) / beta_hat), 0)
}
