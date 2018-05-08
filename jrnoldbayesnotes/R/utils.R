.PACKAGENAME <- "jrnold.bayes.notes"  # nolint

#' List included Stan models
#'
#' List the Stan models included in the package.
#'
#' @return A character vector with the names of the all
#'   the models.
#' @family models
#' @export
list_stan_models <- function() {
  dir(system.file("stan", package = .PACKAGENAME))
}

#' Get and compile an included Stan model
#'
#' Get and compile an Stan model included in the package.
#'
#' @param modelname The name of the model to get.
#' @param ... Passed to \code{\link[rstan]{stan_model}}
#' @return An object of class \code{"\linkS4class{stanmodel}"}.
#' @family models
#' @export
get_stan_model <- function(modelname, ...) {
  if (!stringr::str_detect(modelname, "\\.stan$")) {
    modelname <- stringr::str_c(modelname, ".stan")
  }
  path <- system.file("stan", modelname, package = .PACKAGENAME)
  if (!file.exists(path)) {
    stop(sQuote(modelname), " does not exist. ",
         "Run `list_stan_models` to see the available models.",
         call. = FALSE)
  }
  rstan::stan_model(file = path, ...)
}
