# compare concordance index from R survival package
library(survival)

t_ <- c(1, 2, 3, 4)
s  <- c(4, 3, 2, 1)           # higher risk earlier -> perfect
e  <- rep(TRUE, 4)

concordance(Surv(t_,e)~s, reverse=T)$concordance

t_ <- c(1, 2, 3, 4)
s  <- c(1, 2, 3, 4)           # lower risk earlier -> worst
e  <- rep(TRUE, 4)

concordance(Surv(t_,e)~s, reverse=T)$concordance

# has ties
# Pairs: (0,1)=0.5; (0,2)=1; (1,2)=1  => 2.5/3
t_ <- c(1, 2, 3)
s  <- c(1, 1, 0)
e  <- rep(TRUE, 3)

concordance(Surv(t_,e)~s, reverse=T)$concordance

# First is censored at time 2 -> cannot define pairs
# Only i=2 (event at 3) vs j=3 (time 4): s[2]=0.2 > s[3]=0.1 -> concordant
t_ <- c(2, 3, 4)
e  <- c(FALSE, TRUE, TRUE)
s  <- c(0.9, 0.2, 0.1)

concordance(Surv(t_,e)~s, reverse=T)$concordance


"no comparable pairs returns NaN"
t_ <- c(1, 1)
s  <- c(0.5, 0.7)
e  <- c(FALSE, FALSE)         # all censored

concordance(Surv(t_,e)~s, reverse=T)$concordance
