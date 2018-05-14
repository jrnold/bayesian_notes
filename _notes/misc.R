step_autoscale <- function(
  recipe, ...,
  role = NA,
  trained = FALSE,
  skip = FALSE
) {
  ## The variable selectors are not immediately evaluated by using
  ## the `quos` function in `rlang`
  terms <- rlang::quos(...)
  if(length(terms) == 0)
    stop("Please supply at least one variable specification. See ?selections.")
  add_step(
    recipe,
    step_autoscale_new(
      terms = terms,
      trained = trained,
      role = role,
      skip = skip))
}

step_autoscale_new <- function(
  terms = NULL,
  role = NA,
  trained = FALSE,
  skip = FALSE
) {
  step(
    subclass = "autoscale",
    terms = terms,
    role = role,
    trained = trained,
    skip = skip
  )
}
