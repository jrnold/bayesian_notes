# Create data/alexseev.rda
suppressPackageStartupMessages({
  library("haven")
})

alexseev <- read_dta(here::here("data-raw/alexseev.dta")) %>%
  mutate(region = as.integer(region),
         brdcont = as.logical(brdcont)) %>%
  map_dfc(~ rlang::set_attrs(.x, label = NULL, format.stata = NULL)) %>%
  select(-slavicshare_changenonslav)
usethis::use_data(alexseev, overwrite = TRUE)
