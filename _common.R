suppressPackageStartupMessages({
  library("knitr")
  library("rstan")
  library("tidyverse")
  library("rubbish")
})

set.seed(92508117 )
options(digits = 3)

knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  cache = TRUE,
  out.width = "70%",
  fig.align = 'center',
  fig.width = 6,
  fig.asp = 0.618,  # 1 / phi
  fig.show = "hold"
)

options(dplyr.print_min = 6,
        dplyr.print_max = 6)

# Helpful Documentation functions
rpkg_url <- function(pkg) {
  paste0("https://cran.r-project.org/package=", pkg)
}

rpkg <- function(pkg) {
  paste0("**[", pkg, "](", rpkg_url(pkg), ")**")
}

rdoc_url <- function(pkg, fun) {
  paste0("https://www.rdocumentation.org/packages/", pkg, "/topics/", fun)
}

rdoc <- function(pkg, fun, full_name = FALSE) {
  text <- if (full_name) paste0(pkg, "::", fun) else pkg
  paste0("[", text, "](", rdoc_url(pkg, fun), ")")
}

STAN_VERSION <- "2.14.0"
STAN_URL <- "http://mc-stan.org/documentation/"
STAN_MAN_URL <- paste0("https://github.com/stan-dev/stan/releases/download/v", STAN_VERSION, "/stan-reference-", STAN_VERSION, ".pdf")

standoc <- function(x = NULL) {
  if (!is.null(x)) {
    STAN_MAN_URL
  } else {
    paste("[", x, "](", STAN_MAN_URL, ")")
  }
}

preprocess_lm <- function(formula, data = NULL, weights = NULL,
                          contrasts = NULL, na.action = options("na.action"),
                          offset = NULL, ...) {
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "weights", "na.action",
               "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  mt <- attr(mf, "terms")
  out <- list(
    y = model.response(mf),
    w =  as.vector(model.weights(mf)),
    offset = as.vector(model.offset(mf)),
    X = model.matrix(mt, mf, contrasts),
    terms = mt,
    xlevels = stats::.getXlevels(mt, mf)
  )
  out$n <- nrow(out$X)
  out$k <- ncol(out$X)
  out
}

# placeholder for maybe linking directly to docs
stanfunc <- function(x) {
  paste0("`", x, "`")
}

knit_print.stanmodel <- function(x, options) {
  code_str <- x@model_code
  knitr::asis_output(htmltools::tags$pre(htmltools::tags$code(htmltools::HTML(code_str), class = "stan")))
}


augment.loo <- function(x, data = NULL, ...) {
  out <- as_tibble(x$pointwise)
  names(out) <- paste0(".", names(out))
  out$.pareto_k <- x$pareto_k
  if (!is.null(data)) {
    bind_cols(data, out)
  } else {
    out
  }
}

glance.loo <- function(x, ...) {
  out <- tibble::as_tibble(unclass(x[setdiff(names(x), c("pointwise", "pareto_k"))]))
  out$n <- attr(x, "log_lik_dim")[2]
  out$n_sims <- attr(x, "log_lik_dim")[1]
  out
}

tidy.loo <- function(x, ...) {
  elpd <- grep("^elpd_(loo|waic)$", names(x), value = TRUE)
  p <- grep("^p_(loo|waic)$", names(x), value = TRUE)
  ic <- grep("^(waic|looic)$", names(x), value = TRUE)
  tibble::tibble(
    param = c(elpd, p, ic),
    estimate = c(x[[elpd]], x[[p]], x[[ic]]),
    std.err = c(x[[paste0("se_", elpd)]],
                x[[paste0("se_", p)]],
                x[[paste0("se_", ic)]]))
}

stan_iter <- function(object, permuted = FALSE, inc_warmup = FALSE) {
  if (rstan:::is.stanreg(object)) {
    object <- object$stanfit
  }
  pars <- rstan::extract(object, permuted = permuted,
                         inc_warmup = inc_warmup)
  nchains <- ncol(pars)
  skeleton <- rstan:::create_skeleton(object@model_pars, object@par_dims)
  map(array_branch(pars, 1:2), function(theta) {
    rstan:::rstan_relist(theta, skeleton)
  })
}
