# R/alacoxIC-wrapper.R
#
# Convenience wrapper around ALassoSurvIC::alacoxIC that assembles every
# input required by the post-selection pipeline:
#
#   * betahat  = fit$b   (adaptive lasso estimate, un-normalized scale)
#   * lambda   = the per-coordinate penalty vector theta / |beta_init|
#   * U, jumps = support set and hazard jumps of the estimated baseline
#   * n        = sample size
#
# Why the per-coordinate lambda matters
# -------------------------------------
# ALassoSurvIC minimizes the *adaptive* lasso objective, i.e., the penalty
# is sum_j theta * |beta_j| / |beta_init_j|, where beta_init is the
# unpenalized NPMLE returned in fit$unpen.b. The KKT conditions therefore
# involve the per-coordinate penalty
#
#     lambda_j = theta / |beta_init_j|,
#
# and it is THIS vector (not the scalar theta) that PostSelectIC() expects.
#
# Note on scales: the function always fits with normalize.X = FALSE so that
# betahat, the information matrices and the intervals refer to the original
# covariate scale, matching the simulation setting of the paper. Fitting
# with normalize.X = TRUE and feeding the normalized-scale quantities into
# PostSelectIC() would silently produce wrong intervals.
#
# License: GPL-3

#' Fit the adaptive lasso and assemble inputs for post-selection inference
#'
#' Runs \code{\link[ALassoSurvIC]{alacoxIC}} with \code{normalize.X = FALSE}
#' and extracts everything the post-selection pipeline needs.
#'
#' @param lowerIC,upperIC Numeric vectors: interval endpoints; use
#'   \code{Inf} for right-censored subjects.
#' @param X Covariate matrix (n x p); column names are used to label
#'   results.
#' @param theta Optional scalar: the adaptive lasso tuning parameter.
#'   If omitted (default), \code{alacoxIC} chooses it by BIC.
#' @param tol,niter Convergence tolerance and iteration cap, passed to
#'   \code{\link[ALassoSurvIC]{alacoxIC}}.
#' @param cl Optional \code{parallel} cluster passed to
#'   \code{\link[ALassoSurvIC]{alacoxIC}}.
#' @param ... Further arguments passed to
#'   \code{\link[ALassoSurvIC]{alacoxIC}}.
#'
#' @return A list with components
#'   \item{betahat}{the adaptive lasso estimate \eqn{\hat\beta} (vector of
#'     length p, zeros for unselected covariates);}
#'   \item{lambda}{the per-coordinate penalty vector
#'     \eqn{\theta / |\hat\beta^{\text{init}}_j|}; entries are \code{Inf}
#'     where the unpenalized initial estimate is exactly zero;}
#'   \item{theta}{the scalar tuning parameter (chosen by BIC if not
#'     supplied);}
#'   \item{unpen.b}{the unpenalized NPMLE initial estimate
#'     \eqn{\hat\beta^{\text{init}}};}
#'   \item{U}{right endpoints of the support set (maximal intersections) of
#'     the estimated baseline cumulative hazard;}
#'   \item{jumps}{the corresponding baseline hazard jumps \eqn{\hat\lambda};}
#'   \item{n}{the sample size;}
#'   \item{fit}{the full \code{alacoxIC} object, for advanced use.}
#'
#' @seealso \code{\link{PostSelectIC}}, \code{\link{info_pres}},
#'   \code{\link{info_spres}}, \code{\link{fun_Hessian_3}}.
#' @references Zhang, J., Li, C., & Weng, H. (2025), Section 2.
#' @importFrom ALassoSurvIC alacoxIC
#' @export
fit_alacoxIC <- function(lowerIC, upperIC, X, theta, tol = 1e-3,
                         niter = 1e5, cl = NULL, ...) {
  ## Fit with normalize.X = FALSE: the paper's inference targets the
  ## original covariate scale, and PostSelectIC() must receive quantities
  ## all on that same scale.
  if (missing(theta)) {
    fit <- ALassoSurvIC::alacoxIC(lowerIC = lowerIC, upperIC = upperIC,
                                  X = X, normalize.X = FALSE,
                                  tol = tol, niter = niter, cl = cl, ...)
  } else {
    fit <- ALassoSurvIC::alacoxIC(lowerIC = lowerIC, upperIC = upperIC,
                                  X = X, theta = theta,
                                  normalize.X = FALSE,
                                  tol = tol, niter = niter, cl = cl, ...)
  }

  betahat <- fit$b
  p0 <- length(betahat)

  ## Per-coordinate adaptive lasso penalty: theta / |beta_init_j|
  unpen.b <- fit$unpen.b
  lambda <- rep(fit$theta, p0) / abs(unpen.b)

  ## Guard against the degenerate all-zero selection (no KKT event exists)
  if (all(betahat == 0)) {
    warning("The adaptive lasso selected no variables; ",
            "post-selection inference is not applicable.")
  }

  out <- list(
    call = match.call(),
    betahat = betahat,
    lambda = lambda,
    theta = fit$theta,
    unpen.b = unpen.b,
    U = fit$lambda.set[, 2],
    jumps = fit$lambda,
    n = fit$n,
    fit = fit
  )
  if (!is.null(colnames(X))) {
    names(out$betahat) <- colnames(X)
    names(out$unpen.b) <- colnames(X)
  }
  out
}
