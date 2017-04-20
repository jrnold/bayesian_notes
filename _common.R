library("knitr")
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

options(dplyr.print_min = 6, dplyr.print_max = 6)

suppressPackageStartupMessages({
  library("rstan")
})

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

#knit_print.stanmodel <- function() {  }
