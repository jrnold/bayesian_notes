# create data/votechoice.rda
suppressPackageStartupMessages({
  library("haven")
  library("tidyverse")
})

col_types <- cols(
  caseid = col_double(),
  ideol7b = col_double(),
  presvote = col_integer(),
  retecon = col_double(),
  white = col_double(),
  female = col_integer(),
  age = col_double(),
  educ1_7 = col_double(),
  income = col_double(),
  partyid = col_double(),
  bushiraq = col_double(),
  exptrnout2 = col_integer()
)
votechoice <- read_tsv("data-raw/Hanmer Kalkan AJPS NES example.tab",
                        col_types = col_types, na = "")
usethis::use_data(votechoice, overwrite = TRUE)
