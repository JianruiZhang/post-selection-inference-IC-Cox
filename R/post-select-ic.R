# R/post-select-ic.R
#
# Post-selection inference for the Cox model with interval-censored data:
# p-values and confidence intervals conditional on the model selected by
# lasso, implementing the pivot (7) and the interval construction of
# Section 2.2 of:
#
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the
#   Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# License: GPL-3

#' Post-selection p-values and confidence intervals for the interval-censored
#' Cox model
#'
#' Performs the post-lasso conditional inference of Section 2.2: for each
#' coordinate of the selected coefficient vector it returns the lasso
#' estimate, the one-step estimator, a two-sided selective p-value for
#' \eqn{H_0: \tilde\theta_{M,j} = 0}, and a \code{1 - alpha} confidence
#' interval for \eqn{\tilde\theta_{M,j}}, all conditional on the selection
#' event \eqn{\{\hat M = M, \hat s_M = s_M\}}.
#'
#' The pivotal quantity is the truncated-Gaussian pivot implemented in
#' \code{\link[selectiveInference]{TG.interval}} and
#' \code{\link[selectiveInference]{TG.pvalue}}; see the paper for the
#' derivation and asymptotic validity.
#'
#' \strong{Units of the arguments.} \code{betahat} and the two information
#' matrices must be on the \emph{same} scale as produced by
#' \code{\link[ALassoSurvIC]{alacoxIC}} with \code{normalize.X = FALSE}
#' (pass \code{normalize.X = FALSE} there, or back-transform first).
#' Internally the function works on the \code{sqrt(n)}-scaled pivot
#' \eqn{\theta_M = n^{1/2} \beta_M} of the paper; the returned columns
#' \code{betahat} and \code{betabar} are back on the original
#' \eqn{\beta} scale.
#'
#' @param betahat Numeric vector: the lasso estimator \eqn{\hat\beta}
#'   (zero entries identify the unselected covariates).
#' @param Hessian.1 \eqn{|M| \times |M|} matrix: the estimated
#'   \strong{Fisher information} \eqn{\hat I_{M,M}} (the negative Hessian of
#'   the profile log likelihood) evaluated at \code{betahat}, used for the
#'   one-step estimator (equation 5 of the paper).
#' @param Hessian.2 \eqn{|M| \times |M|} matrix: a second estimate of
#'   \eqn{\hat I_{M,M}} used in the pivot. May be computed by a different
#'   method than \code{Hessian.1} (e.g., PRES/sPRES for the one-step
#'   estimator and least squares for the pivot, as in the paper's
#'   "PRES+LS" / "sPRES+LS" configurations).
#' @param n Positive integer: the sample size.
#' @param lambda The tuning parameter used by the lasso, either a positive
#'   scalar applied to every coordinate, or a numeric vector of per-
#'   coordinate penalties with the same length as \code{betahat}. It must be
#'   the \strong{per-coordinate} penalty actually used in the lasso fit, i.e.,
#'   \code{theta / |beta_init|} for the adaptive lasso of
#'   \code{\link[ALassoSurvIC]{alacoxIC}}.
#' @param alpha Type-I level; confidence intervals have coverage
#'   \code{1 - alpha}. Default 0.1 (the paper's nominal level is set by the
#'   caller; use 0.05 for 95\% intervals).
#' @param complete Logical. If \code{TRUE}, the inputs \code{betahat},
#'   \code{Hessian.1}, \code{Hessian.2} (and a vector \code{lambda}) are
#'   already restricted to the selected model, i.e., to the nonzero entries
#'   of \code{betahat}. If \code{FALSE} (default), the function restricts
#'   them itself.
#'
#' @return A matrix with one row per selected covariate and columns
#'   \item{betahat}{the lasso estimator \eqn{\hat\beta_M};}
#'   \item{betabar}{the one-step estimator \eqn{\tilde\beta_M}
#'     (equation 5 of the paper, returned on the \eqn{\beta} scale);}
#'   \item{LOWER, UPPER}{the \code{1 - alpha} selective confidence interval
#'     for \eqn{\tilde\theta_{M,j}}, on the \eqn{\beta} scale;}
#'   \item{p-value}{two-sided selective p-value for
#'     \eqn{H_0: \tilde\theta_{M,j} = 0}.}
#'
#' @seealso \code{\link{info_pres}}, \code{\link{info_spres}},
#'   \code{\link{info_ls}} for computing the information matrices.
#' @references Zhang, J., Li, C., & Weng, H. (2025), Section 2.2.
#' @importFrom matrixcalc is.positive.definite
#' @export
PostSelectIC <- function(betahat, Hessian.1, Hessian.2, n, lambda,
                         alpha = 0.1, complete = FALSE) {
  ## ----------------------- input checks -------------------------------
  if (!matrixcalc::is.positive.definite(Hessian.1)) {
    stop("first Hessian is not positive definite")
  }
  if (!matrixcalc::is.positive.definite(Hessian.2)) {
    stop("second Hessian is not positive definite")
  }
  if (length(betahat) != nrow(Hessian.1)) {
    stop("Lasso estimator and first Hessian do not have same dimension")
  }
  if (length(betahat) != nrow(Hessian.2)) {
    stop("Lasso estimator and second Hessian do not have same dimension")
  }
  if (length(lambda) > 1 && length(lambda) != length(betahat)) {
    stop("lambda should be a number, or a vector with same length as betahat")
  }

  ## Restrict everything to the selected model M (nonzero lasso estimates)
  if (!complete) {
    M <- which(betahat != 0)
    betahat <- betahat[M]
    Hessian.1 <- Hessian.1[M, M, drop = FALSE]
    Hessian.2 <- Hessian.2[M, M, drop = FALSE]
    if (length(lambda) > 1) {
      lambda <- lambda[M]
    }
  }

  ## Selection event {M, s_M}: conditioning on the sign vector
  s <- sign(betahat)
  A <- -diag(s)                       # A theta_M <= b encodes the sign event
  InvL_M.1 <- n * solve(Hessian.1)    # (I^{-1}_{M,M})-hat for the one-step estimator
  b <- (A %*% InvL_M.1 %*% (lambda * s)) / sqrt(n)

  ## Scaled one-step estimator theta_bar = n^{1/2} beta_M (equation 5):
  ## beta_M = betahat + (lambda/n) I^{-1} s  (the lasso KKT residual term)
  thetabar <- sqrt(n) * betahat + A %*% b

  ## (I^{-1}_{M,M})-hat for the pivot; may come from a different estimator
  InvL_M.2 <- n * solve(Hessian.2)

  result <- matrix(0, nrow = length(s), ncol = 5)
  colnames(result) <- c("betahat", "betabar", "LOWER", "UPPER", "p-value")
  result[, 1] <- betahat
  result[, 2] <- thetabar / sqrt(n)

  ## Coordinate-wise truncated-Gaussian pivot with gamma = e_j
  for (i in seq_along(s)) {
    eta <- rep(0, length(s))
    eta[i] <- 1 / sqrt(n)             # target: gamma' theta_M = beta_j
    result[i, 3:4] <- selectiveInference::TG.interval(thetabar, A, b, eta, InvL_M.2,
                                                      alpha = alpha)$int
    pv <- selectiveInference::TG.pvalue(thetabar, A, b, eta, InvL_M.2)$pv
    result[i, 5] <- 2 * min(pv, 1 - pv)   # two-sided conditional p-value
  }

  rownames(result) <- names(betahat)
  result
}
