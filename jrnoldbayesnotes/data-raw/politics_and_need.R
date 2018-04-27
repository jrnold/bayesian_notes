suppressPackageStartupMessages({
  library("httr")
})

URL <- "https://github.com/carlislerainey/separation/raw/master/data/politics_and_need.rda"

load(url(URL))
usethis::use_data(politics_and_need, overwrite = TRUE)
