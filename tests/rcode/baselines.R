# establish baseline likelihoods

source("pgwsource.R")
library(survival)
library(purrr)

df <- read.csv("../testdata.csv")
df$deceased <- df$deceased == "True"

fits <- list(
  proxy1 = glm(proxy1 ~ 1, data = df, family = binomial),
  proxy2 = glm(proxy2 ~ 1, data = df, family = binomial),
  tx = glm(tx ~ 1, data = df, family = binomial),
  y = pgwfit(Surv(time, deceased) ~ 1, data = df)
)

lls <- map_dbl(fits, logLik)
