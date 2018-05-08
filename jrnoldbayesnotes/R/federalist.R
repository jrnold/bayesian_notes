#' The Federalist Papers
#'
#' Author information and word counts for \dQuote{The Federalist}
#' papers. The dataset \code{federalist} contains metadata and authorship
#' information about the 85 articles. The dataset \code{federalist_wordcounts}
#' contains word counts for each article of 70 function words used by
#' Mosteller and Wallace (1963) to predict authorship of the articles with
#' disputed authorship.
#'
#' @format A list with with two elements
#' \describe{
#' \item{\code{federalist}}{A data frame with 85 rows and 6 columns.
#'   Each row is a document, and the columns are document metadata.}
#' \item{\code{federalist_wordcounts}}{A data frame with 6035 rows and
#'   4 columns. Each row represents the counts of on word.}
#' }
#'
#' @source These datasets were extracted from the
#' \code{\link[corpus]{federalist}} data included in the \pkg{corpus} package.
#'
#' @references
#' Mosteller, F and Wallace, D. L. (1963).
#' \dQuote{Inference in an authorship problem}
#' \emph{Journal of the American Statistical Association}.
"federalist"
