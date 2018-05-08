suppressPackageStartupMessages({
  library("tibble")
})

eightschools <- tibble(
  test_mean = c(28,  8, -3,  7, -1,  1, 18, 12),
  test_sd = c(15, 10, 16, 11,  9, 11, 10, 18)
)
devtools::use_data(eightschools, overwrite = TRUE)
