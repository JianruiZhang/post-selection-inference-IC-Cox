# R/beta-blocks.R  --  EXPERIMENTAL helper routines
#
# Small, self-contained utilities used by the (experimental) least squares
# information estimator in R/info-ls.R. The main, fully supported pipelines
# in this repository rely only on R/post-select-ic.R, R/info-pres.R and
# R/info-spres.R; you do NOT need anything in this file for those.
#
# Part of the code accompanying:
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the
#   Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# License: GPL-3

## ---------------------------------------------------------------------------
## Sub-interval indicators (pure-R fallbacks)
##
## The ALassoSurvIC package ships compiled (Rcpp) versions of these two
## helpers. The fallbacks below reproduce their semantics in plain R so that
## the experimental code in info_ls() does not rely on package internals.
## ---------------------------------------------------------------------------

# A[i, k] = I( L[i] < u[k] <= R[i] ): the jump of the baseline cumulative
# hazard at u[k] falls inside the censoring interval (L[i], R[i]].
fun_sublr <- function(u, l, r) {
  A <- matrix(0, nrow = length(l), ncol = length(u))
  for (i in seq_along(l)) {
    A[i, which(u > l[i] & u <= r[i])] <- 1
  }
  A
}

# A[i, k] = I( u[k] <= lessthan[i] ): the jump at u[k] occurs at or before
# the threshold lessthan[i].
fun_subless <- function(u, lessthan) {
  A <- matrix(0, nrow = length(lessthan), ncol = length(u))
  for (i in seq_along(lessthan)) {
    A[i, which(u <= lessthan[i])] <- 1
  }
  A
}

## ---------------------------------------------------------------------------
## Sign vector / active set
## ---------------------------------------------------------------------------

# Element-wise sign with exact zeros kept at 0 (avoids R's sign(-0) = -0
# printing quirks and keeps the active-set logic explicit).
fun_sgn <- function(b) {
  b <- as.numeric(b)
  out <- numeric(length(b))
  out[b > 0] <- 1
  out[b < 0] <- -1
  out
}

# Indices of the nonzero entries of b (the selected model M-hat).
fun_active <- function(b) {
  which(as.numeric(b) != 0)
}
