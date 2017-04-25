suppressPackageStartupMessages({
  library("knitr")
  library("rstan")
  library("tidyverse")
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
