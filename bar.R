cat_line <- function (...)  {
  cat(..., "\n", sep = "")
}

print.stanfit_summary <- function(x, n = 10, stats = NULL) {
  stats <- stats %||% colnames(x$summary)
  comment <- sprintf("stanfit_summary: %d paramters, %d chains",
                     nrow(x$summary), length(x$c_summary))
  cat_line(comment)
  print(head(x$summary[ , stats, drop = FALSE], n))
}
