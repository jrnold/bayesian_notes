# Create data/duma.rda
suppressPackageStartupMessages({
  library("tidyverse")
  library("haven")
})

duma <- read_dta(here::here("data-raw/alexseev.dta")) %>%
  mutate(region = as.integer(region),
         brdcont = as.logical(brdcont)) %>%
  filter(!is.na(region)) %>%
  map_dfc(~ rlang::set_attrs(.x, label = NULL, format.stata = NULL)) %>%
  select(-slavicshare_changenonslav)
usethis::use_data(duma, overwrite = TRUE)
