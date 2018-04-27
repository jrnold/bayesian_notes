# create data/econ_growth.tsv
suppressPackageStartupMessages({
  library("tidyverse")
})

col_types <- cols(
  country = col_character(),
  econ_growth = col_double(),
  labor_org = col_double(),
  social_dem = col_double()
)

econ_growth <-
  read_tsv(here::here("data-raw/western1995/econ_growth.tsv"),
           na = "", col_types = col_types)
usethis::use_data(econ_growth, overwrite = TRUE)

