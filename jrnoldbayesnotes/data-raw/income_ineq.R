#' Create data/income_ineq.rda
suppressPackageStartupMessages({
  library("tidyverse")
})

col_types <- cols(
  country = col_character(),
  inequality = col_double(),
  turnout = col_double(),
  energy = col_integer(),
  socialism = col_double()
)
income_ineq <- read_tsv(here::here("data-raw/western1995/income_ineq.tsv"),
                        col_types = col_types, na = "")
usethis::use_data(income_ineq)
