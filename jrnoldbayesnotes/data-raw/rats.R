# Rats data
suppressPackageStartupMessages({
  library("tidyverse")
})

URL <- "https://raw.githubusercontent.com/wiki/stan-dev/rstan/rats.txt"
rats <-
  read_table2(URL,
            col_types = cols(
              day8 = col_integer(),
              day15 = col_integer(),
              day22 = col_integer(),
              day29 = col_integer(),
              day36 = col_integer()
            )) %>%
  mutate(id = row_number()) %>%
  gather(day, growth) %>%
  mutate(day = as.integer(str_extract(day, "\\d+")))
devtools::use_data(rats, overwrite = TRUE)
