#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library("R6")
  library("pandocfilters")
  library("dplyr")
})

pandoc_to_json <- function(file, from = "markdown") {
  args <- sprintf("-f %s -t json %s", from, file)
  out <- system2("pandoc", args, stdout = TRUE)
  jsonlite::fromJSON(out, simplifyVector = FALSE, simplifyDataFrame = FALSE,
                     simplifyMatrix = FALSE)
}

# https://stackoverflow.com/questions/2436688/append-an-object-to-a-list-in-r-in-amortized-constant-time-o1
# https://stackoverflow.com/questions/29461530/efficient-linked-list-ordered-set-in-r/29482211#29482211
ExpandingList <- R6Class("ExpandingList",
  public = list(
    initialize = function() {
      private$data <- rlang::new_environment()
    },
    add = function(val) {
      n <- length(private$data)
      private$data[[as.character(n + 1L)]] <- val
      invisible(self)
    },
    as.list = function() {
      x <- as.list(private$data, sorted = FALSE)
      x[order(as.numeric(names(x)))]
    }
  ),
  private = list(
    data = NULL
  )
)

is_url <- function(x) {
  stringr::str_detect(x, stringr::regex("^(https?|doi):", ignore_case = TRUE))
}

stringify <- function(x, meta) {
  results <- ExpandingList$new()
  go <- function(key, value, ...) {
    if (key %in% c("Str", "MetaString")) {
      if (!is_url(value)) {
        results$add(value)
      }
    } else if (key %in% c("Code", "Math", "RawInline", "Cite")) {
      list()
    }
  }
  x <- astrapply(x, go)
  purrr::flatten_chr(results$as.list())
}

parse_text_md <- function(path, from = "markdown") {
  x <- pandoc_to_json(path, from = from)
  stringr::str_c(stringify(x), collapse = " ")
}

normalize_lang <- function(lang = NULL){
  if (!length(lang) || !nchar(lang)) {
    message(str_c("DESCRIPTION does not contain 'Language' field. ",
                  "Defaulting to 'en-US'."))
    lang <- "en-US"
  }
  if (tolower(lang) == "en" || tolower(lang) == "eng") {
    message("Found ambiguous language 'en'. Defaulting to 'en-US")
    lang <- "en-US"
  }
  if (nchar(lang) == 2) {
    oldlang <- lang
    lang <- paste(tolower(lang), toupper(lang), sep = "_")
    message(sprintf("Found ambiguous language '%s'. Defaulting to '%s'",
                    oldlang, lang))
  }
  lang <- gsub("-", "_", lang, fixed = TRUE)
  parts <- strsplit(lang, "_", fixed = TRUE)[[1]]
  parts[1] <- tolower(parts[1])
  parts[-1] <- toupper(parts[-1])
  paste(parts, collapse = "_")
}

spell_check_pandoc_one <- function(path, dict) {
  text <- parse_text_md(path)
  bad_words <- purrr::flatten_chr(hunspell::hunspell(text, dict = dict))
  out <- tibble::tibble(words = bad_words) %>%
    count(words) %>%
    rename(count = n)
  if (nrow(out) > 0) {
    out[["path"]] <- path
  }
  out
}

spell_check_pandoc <- function(path, ignore = character(), lang = "en_US") {
  stopifnot(is.character(ignore))
  lang <- normalize_lang(lang)
  dict <- hunspell::dictionary(lang, add_words = ignore)
  path <- normalizePath(path, mustWork = TRUE)
  purrr::map_df(sort(path), spell_check_pandoc_one, dict = dict) %>%
    group_by(path, words) %>%
    summarise(count = sum(count)) %>%
    arrange(path, words) %>%
    ungroup() %>%
    mutate(path = basename(path))
}

files <- c(dir(here::here(), pattern = "\\.(Rmd)"),
           here::here("README.md"))
ignore <- readLines(here::here("WORDLIST"))
foo <- spell_check_pandoc(files, ignore = ignore)
