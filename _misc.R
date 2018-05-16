scale2 <- function(x, ...) {
  callGeneric("scale2")
}

scale2.default <- function(x, two_sd = FALSE) {
  x <- as.numeric(x)
  sd_x <- sd(x)
  if (two_sd) {
    sd_x <- 2 * sd_x
  }
  mean_x <- mean(x)
  structure((x - mean_x) / sd_x,
            mean = mean_x,
            spread = sd_x)
}

scale2.logical <- function(x,
  x <- as.numeric(x)
  mean_x = mean(x, na.rm = TRUE)
  structure(x - mean(x), mean = mean_x, spread = 1)
}

