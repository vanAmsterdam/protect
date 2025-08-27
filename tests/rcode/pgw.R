# run from script directory

source("pgwsource.R")
library(survival)

df <- read.csv("../testdata.csv")
df$deceased <- df$deceased == "True"

pgw_fit <- pgwfit(Surv(time, deceased) ~ 1, data = df)
print(pgw_fit$mle)
#                      mle         se         z
# b_(Intercept) -0.6787658 0.13993848 -4.850459
# a_(Intercept)  0.8371022 0.09883253  8.469906
# n_(Intercept) -0.1474001 0.05645491 -2.610935

print(pgw_fit$fit)
#       like      aic      bic
# 1    1 4.224806e-05    0    3 -367.1022 7

pgw_fit2 <- pgwfit(Surv(time, deceased) ~ tx + age, data = df)
print(pgw_fit2$mle)
#                       mle         se          z
# b_(Intercept) -0.61243533 0.18787749 -3.2597590
# b_tx          -0.09193421 0.17646485 -0.5209775
# b_age          0.15705516 0.07684773  2.0437190
# a_(Intercept)  0.83349349 0.09926469  8.3966762
# n_(Intercept) -0.13842867 0.05691890 -2.4320332

print(pgw_fit2$fit)
#   code     maxgrad iter npar      like      aic      bic
# 1    1 0.000165841   15    5 -364.7471 739.4943 755.9859
