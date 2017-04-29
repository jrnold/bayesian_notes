make_scale <- function(x, center = TRUE, scale = TRUE) {
  x_mean <- if (center) mean(x) else 0
  x_sd <- if (scale) sd(x) else 1
  function(x) {
    (x - x_mean) / x_mean
  }
}
