# Create data/unionization
suppressPackageStartupMessages({
  library("tidyverse")
})

col_types <- cols(
  country = col_character(),
  union_density = col_double(),
  left_government = col_double(),
  labor_force_size = col_number(),
  econ_conc = col_double()
)
unionization <-
  read_tsv(here::here("data-raw/western1995/unionization.tsv"),
           col_types = col_types, na = "")
usethis::use_data(unionization, overwrite = TRUE)
