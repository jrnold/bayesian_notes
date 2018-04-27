#' create federalist paper data
suppressPackageStartupMessages({
  library("tidyverse")
  library("tidytext")
  library("corpus")
})
data("federalist", package = "corpus")

federalist_docs <- federalist %>%
  # add a document number
  mutate(number = row_number())

federalist_wordcounts <- term_counts(federalist_docs) %>%
  mutate(number = as.integer(text),
         term = as.character(term)) %>%
  select(-text) %>%
  left_join(select(federalist_docs, number, author), by = "number")

functionwords <- readLines("data-raw/functionwords.txt")

federalist_wordcounts <- federalist_wordcounts %>%
  mutate(term = if_else(term %in% functionwords, term, "OTHER")) %>%
  group_by(number, author, term) %>%
  summarise(count = sum(count)) %>%
  ungroup() %>%
  complete(nesting(number, author), term, fill = list(count = 0L)) %>%
  mutate(count = as.integer(count)) %>%
  arrange(number, term)

federalist <- list(
  docs = select(federalist_docs, -text),
  wordcounts = federalist_wordcounts
)

usethis::use_data(federalist, overwrite = TRUE)
