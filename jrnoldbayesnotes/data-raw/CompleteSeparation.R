library("tibble")
CompleteSeparation <- tribble(
  ~y, ~x1, ~x2,
  0, 1, 3,
  0, 2, 2,
  0, 3, -1,
  0, 3, -1,
  1, 5, 2,
  1, 6, 4,
  1, 10, 1,
  1, 11, 0
)
usethis::use_data(CompleteSeparation, overwrite=TRUE)
