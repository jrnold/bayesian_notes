suppressPackageStartupMessages({
})
# only load stuff that I use outside
"%>%" <- magrittr::"%>%"

set.seed(92508117 )
options(digits = 3)

knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  cache = TRUE,
  out.width = "70%",
  fig.align = "center",
  fig.width = 6,
  fig.asp = 0.618,  # 1 / phi
  fig.show = "hold"
)

options(dplyr.print_min = 6,
        dplyr.print_max = 6)

rstan::rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Helpful Documentation functions
rpkg_url <- function(pkg) {
  stringr::str_c("https://cran.r-project.org/package=", pkg)
}

rpkg <- function(pkg) {
  stringr::str_c("**[", pkg, "](", rpkg_url(pkg), ")**")
}

rdoc_url <- function(pkg, fun) {
  stringr::str_c("https://www.rdocumentation.org/packages/", pkg, "/topics/", fun) # nolint
}

rdoc <- function(pkg, fun, full_name = FALSE) {
  text <- if (full_name) stringr::str_c(pkg, "::", fun) else pkg
  stringr::str_c("[", text, "](", rdoc_url(pkg, fun), ")")
}

STAN_VERSION <- "2.14.0"
STAN_URL <- "http://mc-stan.org/documentation/"
# nolint start
STAN_MAN_URL <- stringr::str_c("https://github.com/stan-dev/stan/releases/download/v",
                               STAN_VERSION, "/stan-reference-",
                               STAN_VERSION, ".pdf")
# nolint end

standoc <- function(x = NULL) {
  if (!is.null(x)) {
    STAN_MAN_URL
  } else {
    stringr::str_c("[", x, "](", STAN_MAN_URL, ")")
  }
}

# placeholder for maybe linking directly to docs
stanfunc <- function(x) {
  stringr::str_c("`", x, "`")
}

knit_print.stanmodel <- function(x, options) {
  code_html <- x@model_code %>%
    htmltools::HTML() %>%
    htmltools::tags$code() %>%
    htmltools::tags$pre(class = "stan")
  knitr::asis_output(code_html)
}

print_stanmodel <- function(path) {
  header <- glue::glue("// {basename(path)}")
  stringr::str_c(c(header, readLines(path)), collapse = "\n") %>%
    htmltools::HTML() %>%
    htmltools::tags$code() %>%
    htmltools::tags$pre(class = "stan")
}
