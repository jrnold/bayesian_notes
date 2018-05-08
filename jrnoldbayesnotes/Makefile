DATASETS = QuasiSeparation.rda CompleteSeparation.rda \
	duma.rda econ_growth.rda income_ineq.rda \
	votechoice.rda unionization.rda federalist.rda \
	politics_and_need.rda rats.rda eightschools.rda

.PHONY: build-data
build-data: $(addprefix data/,$(DATASETS))

data/QuasiSeparation.rda: data-raw/QuasiSeparation.R
	RScript $<

data/CompleteSeparation.rda: data-raw/CompleteSeparation.R
	RScript $<

data/duma.rda: data-raw/duma.R data-raw/alexseev.dta
	RScript $<

data/econ_growth.rda: data-raw/econ_growth.R data-raw/western1995/econ_growth.tsv
	RScript $<

data/unionization.rda: data-raw/unionization.R data-raw/western1995/unionization.tsv
	RScript $<

data/income_ineq.rda: data-raw/income_ineq.R data-raw/western1995/income_ineq.tsv
	RScript $<

data/votechoice.rda: data-raw/votechoice.R data-raw/Hanmer_Kalkan_AJPS_NES_example.dta
	RScript $<

data/federalist.rda: data-raw/federalist.R data-raw/functionwords.txt
	RScript $<

data/politics_and_need.rda: data-raw/politics_and_need.R
	RScript $<

data/rats.rda: data-raw/rats.R
	RScript $<

data/eightschools.rda: data-raw/eightschools.R
	RScript $<
