# install.packages(c("testthat", "pROC"))
library(testthat)
library(pROC)

as_factor01 <- function(x) {
  # event_level = "second" => positive class must be the 2nd level (i.e., "1")
  factor(ifelse(x == 1, "1", "0"), levels = c("0","1"))
}

test_that("AUC unweighted: perfect separation", {
  y <- c(0,0,1,1); s <- c(0.1,0.2,0.8,0.9)
  
  # pROC
  r <- roc(response = y, predictor = s, quiet = T, direction="<")
  expect_equal(as.numeric(auc(r)), 1.0, tolerance = 1e-12)
})

test_that("AUC unweighted: complete misranking", {
  y <- c(0,0,1,1); s <- c(0.9,0.8,0.2,0.1)
  
  r <- roc(y, s, quiet = T, direction="<")
  expect_equal(as.numeric(auc(r)), 0.0, tolerance = 1e-12)
})

test_that("AUC unweighted: classic small example", {
  y <- c(0,0,1,1,0,1)
  s <- c(0.1,0.4,0.35,0.8,0.5,0.6)
  
  r <- roc(y, s, quiet = TRUE, direction="<")
  expect_equal(as.numeric(auc(r)), 0.7777777777777778, tolerance = 1e-12)
})

test_that("AUC with ties averages to 0.5", {
  y <- c(0,1,0,1)
  s <- c(0.5,0.5,0.2,0.2)
  
  r <- roc(y, s, quiet = TRUE, direction="<")
  expect_equal(as.numeric(auc(r)), 0.5, tolerance = 1e-12)
})

# ---- Weighted cases via row replication (integer weights) ----

replicate_by_weight <- function(y, s, w) {
  stopifnot(length(y) == length(s), length(s) == length(w))
  idx <- rep(seq_along(y), times = w)
  list(y = y[idx], s = s[idx])
}

test_that("Weighted case via replication: (0.375)", {
  y <- c(0,1,0,1)
  s <- c(0.5,0.5,0.5,0.1)
  w <- c(1,3,1,1)
  rr <- replicate_by_weight(y, s, w)
  
  r <- roc(rr$y, rr$s, quiet = TRUE, direction="<")
  expect_equal(as.numeric(auc(r)), 0.375, tolerance = 1e-12)
})

test_that("Weighted mixed example via replication: (0.85)", {
  y <- c(0,0,1,1,0,1)
  s <- c(0.1,0.4,0.35,0.8,0.5,0.6)
  w <- c(1,2,1,1,1,3)
  rr <- replicate_by_weight(y, s, w)
  
  r <- roc(rr$y, rr$s, quiet = TRUE, direction="<")
  expect_equal(as.numeric(auc(r)), 0.85, tolerance = 1e-12)
})

# ---- Degenerate label sets ----

test_that("All negatives: pROC errors; ", {
  y <- c(0,0,0,0)
  s <- c(0.1,0.2,0.3,0.4)
  expect_error(roc(y, s, quiet = TRUE, direction="<"))
})

test_that("All positives: pROC errors; ", {
  y <- c(1,1,1,1)
  s <- c(0.1,0.2,0.3,0.4)
  expect_error(roc(y, s, quiet = TRUE, direction="<"))
})
