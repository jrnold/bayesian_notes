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
kappa_lambda_funs <- function(tau = 1, sigma = 1, n = 1) {
  nprec <- n * sigma ^ -2
  list(
    # kappa(lambda)
    fun = function(lambda) {
      1 / (1 + nprec * tau ^ 2 * lambda ^ 2)
    },
    # lambda(kappa)
    inv_scale = function(kappa) {
      sqrt(1 / ((1 + nprec * tau ^ 2) * kappa))
    },
    # d(lambda(kappa)) / d kappa
    dinv_scale = function(kappa) {
      -0.5 * (1 + nprec * tau ^ 2) ^ (-0.5) * kappa ^ (-1.5)
    },
    # lambda^2(kappa)
    inv_var = function(kappa) {
      1 / ((1 + nprec * tau ^ 2) * kappa)
    },
    # d(lambda^2(kappa)) / d kappa
    dinv_var = function(kappa) {
      -(1 + nprec * tau ^ 2) ^ (-1) * kappa ^ (-2)
    },
    # lambda^{-2}(kappa)
    inv_prec = function(kappa) {
      (1 + nprec * tau ^ 2) * kappa
    },
    # d(lambda^{-2}(kappa)) / d kappa
    dinv_prec = function(kappa) {
      (1 + nprec * tau ^ 2)
    }
  )
}

kappa_tau_funs <- function(lambda = 1, sigma = 1, n = 1) {
  nprec <- n * sigma ^ -2
  tau2 <- tau ^ 2
  lambda2 <- lambda ^ 2
  list(
    # kappa(lambda)
    fun = function(lambda) {
      1 / (1 + nprec * tau2 * lambda2)
    },
    # lambda(kappa)
    inv_scale = function(kappa) {
      sqrt(1 / ((1 + nprec * lambda2) * kappa))
    },
    # d(lambda(kappa)) / d kappa
    dinv_scale = function(kappa) {
      -0.5 * (1 + nprec * lambda2) ^ (-0.5) * kappa ^ (-1.5)
    },
    # lambda^2(kappa)
    inv_var = function(kappa) {
      1 / ((1 + nprec * lambda2) * kappa)
    },
    # d(lambda^2(kappa)) / d kappa
    dinv_var = function(kappa) {
      -(1 + nprec * lambda2) ^ (-1) * kappa ^ (-2)
    },
    # lambda^{-2}(kappa)
    inv_prec = function(kappa) {
      (1 + nprec * lambda2) * kappa
    },
    # d(lambda^{-2}(kappa)) / d kappa
    dinv_prec = function(kappa) {
      (1 + nprec * lambda2)
    }
  )
}

kappa_sigma_funs <- function(lambda = 1, tau = 1, n = 1) {
  tau2 <- tau ^ 2
  lambda2 <- lambda ^ 2
  list(
    fun = function(sigma) {
      1 / (1 + n * sigma ^ (-2) * tau2 * lambda2)
    },
    inv_scale = function(kappa) {
      tau * lambda * sqrt(n / (1 - kappa))
    },
    dinv_scale = function(kappa) {
      -0.5 * tau * lambda * sqrt(n) * (1 - kappa) ^ (-3 / 2)
    },
    inv_var = function(kappa) {
      (tau2 * lambda2 * n) / (1 - kappa)
    },
    dinv_var = function(kappa) {
      -(tau2 * lambda2 * n) * kappa ^ (-2)
    },
    inv_prec = function(kappa) {
      (1 - kappa) / (n * tau2 * lambda2)
    },
    jacobian_prec = function(kappa) {
      -1 / (n * tau2 * lambda2)
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

fun_or_rep <- function(f, n) {
  if (is.function(f)) f(n)
  else rep(f, n)
}

#' Simulate priors on shrinkage and number of effective parameters
#'
#' Draw samples from the implied priors on the shrinkage parameters (\code{sim_kappa}) and
#' number of effective parameters (\code{sim_meff}) in hierarchical scale-mixture of normal
#' shrinkage priors.
#'
#' @param lambda Either a number or a function which samples from the
#'   local scale, \eqn{\tau}.
#' @param tau Either a number or a function which samples from the
#'   global scales, \eqn{\lambda}.
#' @param sigma Either a number or a function which samples from the observation
#'   scale, \eqn{\sigma}.
#' @param n The number of observations in the data
#' @param D The number of parameters
#' @param sims The number of samples to draw.
#' @return A numeric vector of samples
#' @references
#' @export
sim_meff <- function(lambda, tau = 1, sigma = 1, D = 1, n = 1,
                     sims = 1000) {
  f <- function(tau, precision) {
    kappa <- 1 / (1 + n * precision * tau ^ 2 * fun_or_rep(lambda, D) ^ 2)
    sum(1 - kappa)
  }
  map2_dbl(fun_or_rep(tau), 1 / fun_or_rep(sigma) ^ 2, f)
}

#' @rdname sim_meff
#' @export
sim_kappa <- function(lambda, tau = 1, sigma = 1,
                      n = 1, sims = 1000) {
  precision <- 1 / fun_or_rep(sigma, sims) ^ 2
  1 / (1 + n * precision * fun_or_rep(tau, sims) ^ 2 *
         fun_or_rep(lambda, sims) ^ 2)
}





