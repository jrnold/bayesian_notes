suppressPackageStartupMessages({
  library("knitr")
  library("rstan")
  library("bayesplot")
  library("loo")
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

# placeholder for maybe linking directly to docs
stanfunc <- function(x) {
  paste0("`", x, "`")
}

knit_print.stanmodel <- function(x, options) {
  code_str <- x@model_code
  knitr::asis_output(as.character(htmltools::tags$pre(htmltools::tags$code(htmltools::HTML(code_str), class = "stan"))))
}
