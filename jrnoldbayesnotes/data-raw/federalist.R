#' create federalist paper data
suppressPackageStartupMessages({
  library("tidyverse")
  library("tidytext")
  library("corpus")
})
data("federalist", package = "corpus")

federalist <- federalist %>%
  # add a document number
  mutate(number = row_number())

federalist_wordcounts <- term_counts(federalist) %>%
  mutate(number = as.integer(text),
         term = as.character(term)) %>%
  select(-text) %>%
  left_join(select(federalist, number, author), by = "number")

functionwords <- readLines("data-raw/functionwords.txt")

federalist_wordcounts <- federalist_wordcounts %>%
  mutate(term = if_else(term %in% functionwords, term, "OTHER")) %>%
  group_by(number, author, term) %>%
  summarise(count = sum(count)) %>%
  ungroup() %>%
  complete(nesting(number, author), term, fill = list(count = 0L)) %>%
  mutate(count = as.integer(count)) %>%
  arrange(number, term)

usethis::use_data(federalist_wordcounts, overwrite = TRUE)

federalist <- select(federalist, -text)
usethis::use_data(federalist, overwrite = TRUE)
