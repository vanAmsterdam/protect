######################################################################
## Basic APGW functions used within the optimization

pgwH0 <- function(ti,kap){
   out <- ((kap+1)/kap)*((1+ti/(kap+1))^kap-1)
   k0 <- kap==0
   out[k0] <- log(1+ti)[k0]
   kInf <- kap==Inf
   out[kInf] <- (exp(ti) - 1)[kInf]
   out
}

pgwH0k <- function(ti,kap){
   H0i <- pgwH0(ti,kap)
   h0i <- pgwh0(ti,kap)
   m0i <- pgwm0(ti,kap)
   out <- ((kap*H0i+kap+1)*m0i)/(kap*(kap-1)) -
            H0i/(kap*(kap+1)) - ti*h0i/(kap+1)
   k0 <- kap==0
   out[k0] <- 0
   kInf <- kap==Inf
   out[kInf] <- 0
   out
}

pgwH0kk <- function(ti,kap){
   H0i <- pgwH0(ti,kap)
   H0ki <- pgwH0k(ti,kap)
   h0i <- pgwh0(ti,kap)
   h0ki <- pgwh0k(ti,kap)
   m0i <- pgwm0(ti,kap)
   m0ki <- pgwm0k(ti,kap)
   A <- H0i + kap*H0ki + 1
   B <- kap*H0i + kap + 1
   out <- kap*(kap-1)*(A*m0i+B*m0ki)/((kap*(kap-1))^2) -
          (2*kap-1)*B*m0i/((kap*(kap-1))^2) -
          (kap*(kap+1)*H0ki - (2*kap+1)*H0i)/((kap*(kap+1))^2) -
          ti*((kap+1)*h0ki - h0i)/((kap+1)^2)
   k0 <- kap==0
   out[k0] <- 0
   kInf <- kap==Inf
   out[kInf] <- 0
   out
}

pgwh0 <- function(ti,kap){
   out <- (1+ti/(kap+1))^(kap-1)
   #k0 <- kap==0
   #out[k0] <- 1/(1+ti)[k0]
   kInf <- kap==Inf
   out[kInf] <- exp(ti)[kInf]
   out
}

pgwh0d <- function(ti,kap){
   out <- pgwh0(ti,kap)*pgwm0d(ti,kap)
   out
}

pgwh0k <- function(ti,kap){
   out <- pgwh0(ti,kap)*pgwm0k(ti,kap)
   out
}

pgwm0 <- function(ti,kap){
   out <- (kap-1)*log(1+ti/(kap+1))
   #k0 <- kap==0
   #out[k0] <- -log(1+ti)[k0]
   kInf <- kap==Inf
   out[kInf] <- ti[kInf]
   out
}

pgwm0d <- function(ti,kap){
   out <- (kap-1)/(kap+1+ti)
   #k0 <- kap==0
   #out[k0] <- -1/(1+ti)[k0]
   kInf <- kap==Inf
   out[kInf] <- 1
   out
}

pgwm0dd <- function(ti,kap){
   out <- -(kap-1)/((kap+1+ti)^2)
   #k0 <- kap==0
   #out[k0] <- 1/((1+ti)^2)[k0]
   kInf <- kap==Inf
   out[kInf] <- 0
   out
}

pgwm0k <- function(ti,kap){
   out <-  pgwm0(ti,kap)/(kap-1) - ti*pgwm0d(ti,kap)/(kap+1)
   k0 <- kap==0
   out[k0] <- 0
   kInf <- kap==Inf
   out[kInf] <- 0
   out
}

pgwm0kk <- function(ti,kap){
   m0i <- pgwm0(ti,kap)
   m0ki <- pgwm0k(ti,kap)
   m0di <- pgwm0d(ti,kap)
   m0kdi <- pgwm0kd(ti,kap)
   out <- ((kap-1)*m0ki-m0i)/((kap-1)^2) -
          ti*((kap+1)*m0kdi-m0di)/((kap+1)^2)
   k0 <- kap==0
   out[k0] <- 0
   kInf <- kap==Inf
   out[kInf] <- 0
   out
}

pgwm0kd <- function(ti,kap){
   out <- (2+ti)/((kap+1+ti)^2)
   k0 <- kap==0
   out[k0] <- 0
   kInf <- kap==Inf
   out[kInf] <- 0
   out
}

pgwsim <- function(param, u){
   phi <- exp(param[,1])
   lam <- exp(param[,2])
   gam <- exp(param[,3])
   kap <- exp(param[,4])-1
   (1/phi)*((pgwH0(-log(u)/lam, 1/kap))^(1/gam))
}

pgwS <- function(param, t){
   phi <- exp(param[,1])
   lam <- exp(param[,2])
   gam <- exp(param[,3])
   kap <- exp(param[,4])-1
   exp(-lam*pgwH0((phi*t)^gam, kap))
}

pgwh <- function(param, t){
   phi <- exp(param[,1])
   lam <- exp(param[,2])
   gam <- exp(param[,3])
   kap <- exp(param[,4])-1
   pgwh0((phi*t)^gam, kap)*lam*gam*phi*((phi*t)^(gam-1))
}




######################################################################
## Function: pgwfit
## pgwfit is used to fit the APGW model to data (demonstrations can be found
## at the end of this script) which, by default, uses the R's inbuilt nlm
## optimizer to minimize the negative likelihood, i.e., maximize likelihood.
## Alternatively, by setting usenlm=FALSE, our own Newton-Raphson procedure is
## used. This seems to be faster than nlm but, of course, is much simpler,
## and therefore, likely to be less stable by comparison with nlm.
## Note: In any case (when goodinit=TRUE) we use nlm to produce "good"
## initial values by fitting a "null" model (no covariates).

pgwfit <- function(formula, data=NULL, fix, init, usenlm=TRUE, goodinit=TRUE,
                     hessian=TRUE, iterlim=1000, ...){

   data0 <- data
   mf <- model.frame(formula, data)
   tideltai <- model.extract(mf, "response")[, 1:2]
   X <- model.matrix(formula, mf)
   
   # keep varnames
   varnames <- colnames(X)
   
   k1 <- dim(X)[2]
   k <- rep(k1,3)

   ## data now in form required for optimizer
   data <- cbind(tideltai, X, X, X, X)

   # parnam <- c(paste("t", 0:(k1-1), sep=""), 
   #             paste("b", 0:(k1-1), sep=""),
   #             paste("a", 0:(k1-1), sep=""), 
   #             paste("n", 0:(k1-1), sep="") )
   parnam <- c(paste("t", varnames, sep='_'),
               paste("b", varnames, sep='_'),
               paste("a", varnames, sep='_'),
               paste("n", varnames, sep='_'))

   if(missing(fix)){ ## fits PH model by default
               ## tau           beta            alpha           nu
      fix <- c(0,rep(0,k1-1), NA,rep(NA,k1-1), NA,rep(0,k1-1),  NA,rep(0,k1-1))
   }

   whichpar <- is.na(fix)

   if(missing(init)){
      t00 <- b00 <- log(sum(tideltai[,2])/sum(tideltai[,1]))
      init <- c(t00,rep(0.01,k1-1),b00,rep(0.01,k1-1),
                0.01,rep(0.01,k1-1),0.4,rep(0.01,k1-1))
      if(goodinit){ 
         ## "good" initial values by fitting a null model (i.e., no covariaes)
         ## using nlm
         fix0 <- matrix(fix, byrow=TRUE, nrow=4)[,1]
         init0 <- c(t00,b00,0.01,0.4)
         form0 <- update.formula(formula, .~1)
         k0 <- rep(1,3)
         m0 <- pgwfit(form0, data0, fix0, init0, usenlm=TRUE,
                        hessian=FALSE, iterlim=iterlim)
         init0[is.na(fix0)] <- m0$mle[,1]
         init[seq(1,length=4,by=k1)] <- init0
      }
   }

   if(usenlm){
      fit <- nlm(loglikepgw, init, data=data, k=k, fix=fix,
                     hessian=hessian, iterlim=iterlim, ...)
   }else{
      fit <- pgwNR(param=init, data=data, k=k, fix=fix, iterlim=iterlim, ...)
   }

   code <- fit$code

   mle1 <- fit$est
   mle1[!whichpar] <- fix[!whichpar]
   mlemat <- matrix(mle1, byrow=T, nrow=4)

   mle <- mle1[whichpar]
   
   hess <- fit$hess[whichpar,whichpar] ## information matrix as it is hessian
                                       ## for negative log-likelihood
   if(is.null(hess)){
      hess <- NA
   }else{
      hess <- as.matrix(hess)
      rownames(hess) <- colnames(hess) <- parnam[whichpar]
   }

   gradient <- fit$gradient[whichpar]
   maxgrad <- max(abs(gradient))
   covmat <- solve(hess)
   se <- sqrt(diag(covmat))
   mle <- cbind(mle, se, z=mle/se)
   rownames(mle) <- parnam[whichpar]

   npar <- sum(whichpar)
   like <- -fit$min
   iter <- fit$iter
   aic <- -2*like + 2*npar
   bic <- -2*like + log(dim(data)[1])*npar

   fit <- data.frame(code=code, maxgrad=maxgrad, iter=iter, npar=npar,
                     like=like, aic=aic, bic=bic)

   out <- list(fit=fit, mle=mle, fix=fix, grad=gradient, vcov=covmat)

   # add class to be able to define methods
   class(out) <- c('pgwfit', class(out))
   
   # add formula for predictions on new data
   out[['formula']] <- formula
   
   out
}


lpfun <- function(phi, lam, gam, kap, ti, di) {
# re-implementation of loglike function for easier export
   zi <- (phi*ti)^gam
   h0i <- pgwh0(zi, kap)
   H0i <- pgwH0(zi, kap)
   di*log((lam*gam*zi/ti)*h0i) - lam*H0i
}

lpfun_logprm <- function(tau, bet, alp, nu, ti, di) {
# re-implementation of loglike function for easier export using log params
   phi <- exp(tau)
   lam <- exp(bet)
   gam <- exp(alp)
   kap <- exp(nu)-1
  
   return(lpfun(phi, lam, gam, kap, ti, di)) 
}

# define the regularization term
elastic_reg <- function(beta, lambda = 0, alpha = 0) {
   ridge <- (1 - alpha) * sum(beta^2)
   lasso <- alpha * p_norm(beta, 1)
   return(lambda * (lasso + ridge))
}


######################################################################
## Function: loglikepgw
## Likelihood function which is supplied to either the nlm optimizer or our
## own Newton-Raphson optimizer (the pgwNR below).
## Note the two arguments "deriv1" and "deriv2" which, when set to TRUE,
## respectively correpsond to the calculation of analytic first and second
## derivatives. When using nlm, we recommend using the analytic first
## derivatives to speed up the procedure but letting nlm work out the second
## derivatives numerically. Note that deriv1=TRUE and deriv1=FALSE are indeed
## the defaults.
## When the likelihood function is maximized using our own pgwNR procedure,
## the the analytic first and second derivatives are used. As mentioned above,
## our pgwNR procedure is faster than nlm, but is likely to be less stable
## in general as nlm is a more advanced optimizer.

loglikepgw <- function(param, data, k, fix, deriv1=TRUE, deriv2=FALSE,
                       lambda=0, alpha=0){

   p <- length(param)
   q <- dim(data)[2]

   if(missing(fix)){
      fix <- rep(NA, p)
   }

   whichfix <- !is.na(fix)
   param[whichfix] <- fix[whichfix]

   k1 <- k[1]
   k2 <- k[2]
   k3 <- k[3]

   tau <- param[1:k1]
   beta <- param[(k1+1):(k1+k2)]
   alpha <- param[(k1+k2+1):(k1+k2+k3)]
   nu <- param[(k1+k2+k3+1):p]

   ti <- data[,1]
   di <- data[,2]

   Xt <- as.matrix(data[,3:(k1+2)])
   Xb <- as.matrix(data[,(k1+3):(k1+k2+2)])
   Xa <- as.matrix(data[,(k1+k2+3):(k1+k2+k3+2)])
   Xn <- as.matrix(data[,(k1+k2+k3+3):q])

   xti <- Xt%*%tau
   xbi <- Xb%*%beta
   xai <- Xa%*%alpha
   xni <- Xn%*%nu

   phi <- exp(xti)
   lam <- exp(xbi)
   gam <- exp(xai)
   kap <- exp(xni)-1

   zi <- (phi*ti)^gam
   h0i <- pgwh0(zi, kap)
   H0i <- pgwH0(zi, kap)
   H0ki <- pgwH0k(zi,kap)
   m0i <- pgwm0(zi, kap)
   m0di <- pgwm0d(zi, kap)
   m0ki <- pgwm0k(zi,kap)

   lzhi <- lam*zi*h0i
   logzi <- log(zi)
   zlogzi <- zi*logzi
   kp1 <- kap + 1

   # loglike <- -sum( di*log((lam*gam*zi/ti)*h0i) - lam*H0i )
   loglike <- -sum(lpfun(phi,lam,gam,kap,ti,di))
   
   if(deriv1){
      dldtau <- t( gam*(di*(1+zi*m0di) - lzhi) )%*%Xt
      dldbet <- t( di - lam*H0i )%*%Xb
      dldalp <- t( di*(1 + logzi + zlogzi*m0di) - lzhi*logzi )%*%Xa
      dldnu <- t( kp1*(di*m0ki - lam*H0ki) )%*%Xn

      grad <- -c(dldtau, dldbet, dldalp, dldnu)
      grad[whichfix] <- 0

      attr(loglike, "gradient") <- grad
   }

   if(deriv2){
      H0kki <- pgwH0kk(zi,kap)
      h0di <- pgwh0d(zi,kap)
      h0ki <- pgwh0k(zi,kap)
      m0ddi <- pgwm0dd(zi,kap)
      m0kki <- pgwm0kk(zi,kap)
      m0kdi <- pgwm0kd(zi,kap)

      Wtt <- (gam^2)*zi*(di*(m0di+zi*m0ddi)-lam*(h0i+zi*h0di))
      Wtb <- -gam*lzhi
      Wta <- gam*(di*(1+zi*m0di)-lzhi+logzi*Wtt/(gam^2))
      Wtn <- kp1*gam*zi*(di*m0kdi-lam*h0ki)

      Wbb <- -lam*H0i
      Wba <- -lzhi*logzi
      Wbn <- -lam*kp1*H0ki

      Waa <- zlogzi*(di/zi + (1+logzi)*(di*m0di-lam*h0i) + zlogzi*(di*m0ddi-lam*h0di))
      Wan <- logzi*Wtn/gam

      Wnn <- kp1*(di*m0ki-lam*H0ki + kp1*(di*m0kki-lam*H0kki))

      Htt <- t(Xt*c(Wtt))%*%Xt
      Htb <- t(Xt*c(Wtb))%*%Xb
      Hta <- t(Xt*c(Wta))%*%Xa
      Htn <- t(Xt*c(Wtn))%*%Xn

      Hbt <- t(Htb)
      Hbb <- t(Xb*c(Wbb))%*%Xb
      Hba <- t(Xb*c(Wba))%*%Xa
      Hbn <- t(Xb*c(Wbn))%*%Xn

      Hat <- t(Hta)
      Hab <- t(Hba)
      Haa <- t(Xa*c(Waa))%*%Xa
      Han <- t(Xa*c(Wan))%*%Xn

      Hnt <- t(Htn)
      Hnb <- t(Hbn)
      Hna <- t(Han)
      Hnn <- t(Xn*c(Wnn))%*%Xn

      Hrt <- cbind(Htt,Htb,Hta,Htn)
      Hrb <- cbind(Hbt,Hbb,Hba,Hbn)
      Hra <- cbind(Hat,Hab,Haa,Han)
      Hrn <- cbind(Hnt,Hnb,Hna,Hnn)

      H <- rbind(Hrt,Hrb,Hra,Hrn)
      H[whichfix,] <- 0
      H[,whichfix] <- 0

      attr(loglike, "hessian") <- -H
   }

   loglike
}




######################################################################
## Function: pgwNR
## Our implementation of the Newton-Raphson procedure (which makes use of
## step-halving) in order to minimize the negative likelihood, i.e., maximize
## the likelihood.

pgwNR <- function(param, data, k, fix, iterlim=1000, tol=1e-4, halfmax=100){

   paramold <- Inf
   iter <- 0

   while(max(abs(param - paramold)) > tol & iter <= iterlim){

      iter <- iter + 1
      paramold <- param

      whichfix <- !is.na(fix)
      param[whichfix] <- fix[whichfix]
      whichpar <- !whichfix

      loglike <- loglikepgw(param=param, data=data, k=k, fix=fix,
                    deriv1=TRUE, deriv2=TRUE)

      grad <- attr(loglike,"gradient")
      negH <- -attr(loglike,"hessian")

      dp <- solve(negH[whichpar,whichpar], grad[whichpar])

      j <- 0
      del <- 1

      loglikeold <- loglike
      loglike <- Inf

      ## step-halving
      while(loglike > loglikeold & j < halfmax){

         del <- del/(2^j)
         param[whichpar] <- paramold[whichpar] + del*dp

         loglike <- loglikepgw(param=param, data=data, k=k, fix=fix,
                                   deriv1=FALSE, deriv2=FALSE)

         loglike <- ifelse(is.na(loglike), Inf, loglike)
         j <- j + 1
      }
   }

   loglike <- loglikepgw(param=param, data=data, k=k, fix=fix,
                    deriv1=TRUE, deriv2=TRUE)

   grad <- attr(loglike,"gradient")
   H <- attr(loglike,"hessian")

   code <- 2
   if(iter >= iterlim){
      code <- 4
   }

   list(minimum=loglike, estimate=param, gradient=grad, hessian=H,
        code=code, iterations=iter)
}




######################################################################
## Function: fitvec
## fitvec is a convenience function used below when calculating the estimated
## survivor curve for a given level of a categorical covariate

fitvec <- function(fit, grp=1){
   mlevec <- fit$fix
   mlevec[is.na(mlevec)] <- fit$mle[,1]
   mlemat <- matrix(mlevec, byrow=T, nrow=4)
   mlemat[is.na(mlemat[,1]),1] <- fit$mle[,1]
   if(grp==1){
      mlemat[,1]
   }else{
      mlemat[,1]+mlemat[,grp]
   }
}

######################################################################
#' grab coefficients from pgwfit class

coef.pgwfit <- function(fit) fit$mle[, 'mle']
vcov.pgwfit <- function(fit) fit$vcov
se.pgwfit   <- function(fit) fit$mle[, 'se']


#####
#' Get logLik from power generalized weibull fit
#' 
#' Get logLik from power generalized weibull fit
#' 
#' @param fit result of call to pgwfit
logLik.pgwfit <- function(fit) as.numeric(fit$fit['like'])

AIC.pgwfit <- function(fit) as.numeric(fit$fit['aic'])
BIC.pgwfit <- function(fit) as.numeric(fit$fit['bic'])

## tidy
tidy.pgwfit <- function(fit, ...) {
   coefs      <- coef(fit)
   std.errors <- se.pgwfit(fit)
   zs         <- fit$mle[, 'z']
   conf.lows  <- coefs - 1.96 * std.errors
   conf.highs <- coefs + 1.96 * std.errors
   out <- dplyr::tibble(
      term      = names(coefs),
      estimate  = coefs,
      std.error = std.errors,
      statistic = zs,
      p.value=NA,conf.low=conf.lows,conf.high=conf.highs
   )
   return(out)
}

## glance, needed for pooling with mice; return as dummy
glance.pgwfit <- function (x, ...) 
{
   ret <- x$fit
   broom::finish_glance(ret, x)
}

find_u <- function(param, t, max_time = 10, eps=1e-6) {
  # given parameters for a power-generalized weibull models
  # and observed time t
  # find the residual 'u' that produces this observation
  stopifnot(t <= max_time)
  fn <- function(u) {
    # generate a time for this u (not that pgw makes high times for low u)
    tsim <- pgwsim(param, u)
    tsim <- ifelse(is.nan(tsim), 2*max_time, tsim)
    return(tsim - t)
  }
  res <- uniroot(fn, c(eps, 1-eps))
  return(res)
}

## predict beta on new data
predict.pgwfit <- function(x, newdata, type="link") {
  if (is.null(x$formula)) stop("object needs formula to create Xmat")
   form <- x$formula
   coefs <- coef(x)
   beta_coefs <- str_subset(names(coefs), '^b_')
   betas <- coefs[beta_coefs]
   if (!(length(beta_coefs) == length(coefs) - 2)) stop(paste('only works when there are no alpha or nu regressions, found coefs:', names(coefs)))
   stopifnot(beta_coefs[1] == 'b_(Intercept)')
   Xmat <- model.matrix(form, data=newdata)
   beta_hat = Xmat %*% betas
   beta_hat = as.vector(beta_hat)
   if (type == "link") {
     return(beta_hat)
   } else if (type == "median") {
     s_fn <- function(b) {
       # function to predict median survival
       pgwsim(cbind(0, b, coefs['a_(Intercept)'], coefs['n_(Intercept)']), 0.5)
     }
     return(s_fn(beta_hat))
   }
}

residuals.pgwfit <- function(x, newdata=NULL, max_time=10, eps=1e-6) {
  # calculate u-residuals on new data
  alpha0 = x$mle['a_(Intercept)','mle']
  nu0    = x$mle['n_(Intercept)','mle']
  if (is.null(newdata)) {
    warning("putting in dummy data, results are meaningless")
    return(0)
  }
  beta_hats <- predict(x, newdata, type='link')
  u_fun <- function(beta, t) {
    param <- cbind(0., beta, alpha0, nu0)
    u_res <- find_u(param, t, max_time, eps)
    return(u_res$root)
  }
  u_funv <- Vectorize(u_fun)
  u_hats <- purrr::map2_dbl(beta_hats, newdata$time, u_funv)
  return(u_hats)
}

## return loglik on new data
pgwfitlogprob <- function(fit, newdata, timevar='time', eventvar='deceased') {
   stopifnot('pgwfit' %in% class(fit))
   coefs <- coef(fit)
   df <- as.data.table(newdata)
   setnames(df, c(timevar, eventvar), c('time', 'deceased'))
   df[, rn:=1:.N]
   df[, a0:=coefs['a_(Intercept)']]
   df[, n0:=coefs['n_(Intercept)']]
   df[, b1:=predict.pgwfit(fit, newdata=.SD), by='rn']
   df[!is.na(b1), ll:=lpfun_logprm(0, b1, a0, n0, time, deceased)]
   return(df$ll)
   # if (!is.data.table(newdata)) {
   #    df <- as.data.table(newdata)
   # } else{
   #    df <- newdata
   # }
   # if (!(timevar == 'time' & eventvar=='deceased')) setnames(df, c(timevar, eventvar), c('time', 'deceased'))
   # # df[, rn:=1:.N]
   # a0=coefs['a_(Intercept)']
   # n0=coefs['n_(Intercept)']
   # b1s <- df[, list(b1=predict(fit, newdata=.SD)), by='rn']
   # lls <- df[, ifelse(is.na(b1s$b1), NA, lpfun_logprm(0, b1s$b1, a0, n0, time, deceased))]
   # return(df$ll)
}

## make pgwfit objects
make_pgwfit <- function(fit=NULL, mle=NULL, fix=NULL, grad=NULL, vcov=NULL, formula=NULL,
                        coefficients=NULL) {
  if (!is.null(coefficients)) {
    stopifnot(is.null(mle))
    mle <- as.matrix(coefficients)[, c('estimate', 'std.error', 'statistic')]
    rownames(mle) <- coefficients$term
    colnames(mle) <- c('mle', 'se', 'z')
    class(mle) <- 'numeric'
  }
  out <- list(fit=fit,mle=mle,fix=fix,grad=grad,vcov=vcov,formula=formula)
  class(out) <- c('pgwfit', class(out))
  return(out)
}

## survival function utils
make_marginal_surv_fn <- function(fit) {
  # for pgwfit with only intercepts, make survival function
  stopifnot(nrow(fit$mle) == 3)
  beta0  = fit$mle['b_(Intercept)','mle']
  alpha0 = fit$mle['a_(Intercept)','mle']
  nu0    = fit$mle['n_(Intercept)','mle']
  sfn <- function(t) pgwS(cbind(0., beta0, alpha0, nu0), t)
  return(sfn)
}

make_marginalized_surv_fn <- function(fit, data) {
  #' make a function that predicts the survival marginalized over observed covariate distribution
  beta_hats <- predict(fit, newdata=data, type='link')
  alpha0 = fit$mle['a_(Intercept)','mle']
  nu0    = fit$mle['n_(Intercept)','mle']
  sfn <- function(t) {
    shats <- sapply(beta_hats, function(beta) pgwS(cbind(0., beta, alpha0, nu0), t))
    return(mean(shats))
  }
  return(Vectorize(sfn))
}

# fit an intercept pgw model using existing fit as offset
fit_b0 <- function(fit, data, nboot=0) {
  if (!is.data.table(data)) setDT(data)
  alpha0=fit$mle['a_(Intercept)','mle']
  nu0=fit$mle['n_(Intercept)','mle']
  
  fit1 <- function(datai) {
    beta_hats=predict(fit, newdata=datai, type='link')
    crit_fn <- function(x) -sum(lpfun_logprm(0.0, x + beta_hats, alpha0, nu0,
                                             datai$time, datai$deceased))
    res <- optimize(crit_fn, interval=c(-10, 10))
    return(res$minimum)
  }
  f0 = fit1(data)
  out = list(b0 = f0, b0_boot=NULL)
  
  if (nboot > 0) {
    bootfn <- function(j) {
      bootiis <- sample(data[,.N], replace=T)
      dataj <- data[bootiis]
      return(fit1(dataj))
    }
    bootres <- map_dbl(1:nboot, bootfn)
    out$b0_boot <- bootres
  }
             
  
  return(out)
  
}
