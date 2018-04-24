# create data/votechoice.rda
suppressPackageStartupMessages({
  library("haven")
  library("tidyverse")
})

votechoice <- read_dta("data-raw/Hanmer Kalkan AJPS NES example.dta") %>%
  mutate_if(is.labelled, as_factor) %>%
  filter(!is.na(exptrnout2)) %>%
  select(-exptrnout2) %>%
  mutate(bushvote = presvote == "Bush",
         caseid = as.integer(caseid),
         retecon = factor(retecon * 2 + 3,
                     labels = c("Much Worse",
                       "Somewhat Worse",
                       "Same",
                       "Somewhat Better",
                       "Much Better")),
         bushiraq = factor(bushiraq * 3 + 1,
                      labels = c("Disapprove strongly",
                                 "Disapprove Not Strongly",
                                 "Approve Not Strongly",
                                 "Approve Strongly")),
         partyid = factor(partyid,
                          labels = c("Strong Democrat", "Weak Democrat", "Independent Democrat",
                            "Independent", "Independent Republican", "Weak Republican",
                            "Republican")),
         educ1_7 = factor(educ1_7,
                          labels = c("0 - 8 Years",
                                     "High School; No Degree",
                                     "High School Degree",
                                     "Some College; No Degree",
                                     "Assoc. Degree",
                                     "College Degree",
                                     "Advanced Degree"))) %>%
  mutate_at(vars(white, female), as.logical) %>%
  select(-presvote)
usethis::use_data(votechoice, overwrite = TRUE)
