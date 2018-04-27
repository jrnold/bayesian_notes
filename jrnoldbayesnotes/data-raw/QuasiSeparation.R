library("tibble")
QuasiSeparation <- tibble::tibble(
  y = c(0, 0, 0, 0, 1, 1, 1, 1, 1, 1),
  x1 = c(1, 2, 3, 3, 3, 4, 5, 6, 10, 11),
  x2 = c(3, 0, -1, 4, 1, 0, 2, 7, 3, 4)
)
usethis::use_data(QuasiSeparation, overwrite = TRUE)
