stanfit_summary <- function(x) {
  structure(x, class = "stanfit_summary")
}

# print.stanfit_summary <- function(x, ..., n = NULL, width = NULL, n_extra = NULL,
#                                   stats = c("n_eff", "Rhat", "mean", "se_mean", "sd")) {
#   x <- as.data.frame(x$summary)
#   x$parameter <- rownames(x)
#   print(tibble::trunc_mat(x[ , c("parameter", stats)], n = n, width = width, n_extra = n_extra))
#   invisible(x)
# }

as.data.frame.stanfit_summary <- function(object, summary = TRUE, chains = TRUE, ...) {
  assert_that(is.flag(summary))
  assert_that(is.flag(chains))
  assert_that(summary || chains)
  out <- vector("list", 2)
  if (summary) {
    x <- as.data.frame(object$summary)
    x <- rownames_to_column(x, "parameter")
    res <- as_tibble(x)
    res$chain <- "all"
    out[[1]] <- res
  }
  if (chains) {
    nm <- dimnames(object$c_summary)
    f <- function(.chain, x) {
      colnames(x) <- nm$stats
      res <- as_data_frame(x)
      res$parameter <- nm$parameter
      res$chain <- .chain
      res
    }
    out[[2]] <- purrr::map2_df(nm$chains, purrr::array_branch(summ$c_summary, 3), f)
  }
  select_(bind_rows(out), ~ chain, ~ parameter, ~ everything())
}
