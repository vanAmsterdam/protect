# run from root directory with `Rscript tests/r_scripts/test_pgw.R`

library(survival)
source("pgwsource.R")

# simulate survival data from power generalized weibull mdoel
n <- 1e5 # number of observations
set.seed(123)

x <- rnorm(n) # covariate
t <- rbinom(n, 1, 0.5)

beta0 <- -2.3
beta_x <- 1
beta_t <- 1
alpha0 <- -0.01
n0 <- 0.7

beta <- beta0 + beta_x * x + beta_t * t
u <- runif(n)
maxtime <- 100

times <- pgwsim(cbind(0.0, beta, alpha0, n0), u)

df <- data.frame(
  time = times,
  t = t,
  deceased = 1
)

df$deceased <- df$time <= maxtime
df$time <- ifelse(df$deceased, df$time, maxtime)

plot(survfit(Surv(time, deceased) ~ 1, data = df), xlab = "Time", ylab = "Survival Probability")
pgw_fit <- pgwfit(Surv(time, deceased) ~ x + t, data = df)
pgw_marg <- pgwfit(Surv(time, deceased) ~ t, data = df)

pgw_fit$mle
# mle          se           z
# b_(Intercept) -2.295557243 0.008458954 -271.376026
# b_x            0.996221635 0.004154561  239.789879
# b_t            0.997981364 0.006882362  145.005656
# a_(Intercept) -0.007920272 0.005199531   -1.523267
# n_(Intercept)  0.696656603 0.004883242  142.662732
pgw_marg$mle
# mle          se          z
# b_(Intercept) -1.60746478 0.007359656 -218.41576
# b_t            0.64244025 0.006538402   98.25646
# a_(Intercept) -0.02641886 0.005488937   -4.81311
# n_(Intercept)  0.38231204 0.003857132   99.11823
