suppressPackageStartupMessages({
  library("httr")
})

URL <- "https://github.com/carlislerainey/separation/raw/master/data/politics_and_need.rda" # nolint

load(url(URL))
usethis::use_data(politics_and_need, overwrite = TRUE)
