# R/info-spres.R
#
# sPRES (simplified PRES) information estimator, Section 2.3.4 of:
#
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the
#   Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# The score of the profile log likelihood admits the closed form
#
#     S~(b~) = n P_n l_beta(b~, Lambda-hat_{b~}),
#
# so the EM re-fit of PRES is only needed to obtain Lambda-hat_{b~} (via
# fun_ew / fun_updatelambda); the score itself is then computed in closed
# form and differentiated by the same Richardson extrapolation as PRES.
#
# License: GPL-3

#' Closed-form profile score S~(b~) used by sPRES
#'
#' Evaluates
#' \deqn{\tilde S(\tilde\beta) = n P_n\, l_\beta(\tilde\beta,\,
#' \hat\Lambda_{\tilde\beta}),}
#' the score of the observed-data log likelihood evaluated at the working
#' coefficient \eqn{\tilde\beta} together with the maximizer
#' \eqn{\hat\Lambda_{\tilde\beta}} of the expected complete-data log
#' likelihood (obtained from the EM updates of \code{ALassoSurvIC}). This is
#' the closed-form counterpart of the quantity computed by [fun_Spsi] in
#' the PRES method.
#'
#' @param arglist A list as produced by
#'   \code{\link[ALassoSurvIC]{fun_arglist}}, additionally carrying
#'   \code{initial_lambda}.
#' @param b Numeric vector: the working coefficient value \eqn{\tilde\beta}.
#' @param add Small positive constant: for right-censored subjects
#'   (\code{R = Inf}) the effective interval end is \code{R + add}.
#'
#' @return Numeric vector of length \code{length(b)}: the profile score.
#'
#' @keywords internal
#' @noRd
fun_Spsi_2 <- function(arglist, b, add = 1e-5) {
  n <- arglist$n
  Z <- arglist$z
  L <- arglist$l
  R <- arglist$r
  U <- arglist$set[, 2]
  m <- length(U)

  ## EM updates for the baseline hazard jumps at the working value b
  distance <- arglist$tol + 1000
  iter <- 1
  old_lambda <- arglist$initial_lambda
  while ((distance > arglist$tol) & (iter < arglist$niter)) {
    ew <- ALassoSurvIC::fun_ew(b, old_lambda, arglist)
    new_lambda <- ALassoSurvIC::fun_updatelambda(b, ew, arglist)
    distance <- max(abs(new_lambda - old_lambda))
    old_lambda <- new_lambda
    iter <- iter + 1
  }
  lambda <- new_lambda

  ## Baseline cumulative hazard at L and at R + add, scaled by exp(Z b)
  exp_bt_Z <- exp(Z %*% b)

  Lambda.L <- Lambda.R <- rep(0, n)
  for (i in 1:n) {
    Lambda.L[i] <- sum(lambda[which(U <= L[i])])
    Lambda.R[i] <- sum(lambda[which(U <= (R[i] + add))])
  }

  Lambda.L.exp <- Lambda.L * exp_bt_Z
  Lambda.R.exp <- Lambda.R * exp_bt_Z

  ## Interval probabilities: S(L) - S(R), with S(Inf) = 0
  exp.L <- exp(-Lambda.L.exp)
  exp.R <- exp(-Lambda.R.exp)
  exp.R[which(R == Inf)] <- 0
  denum <- exp.L - exp.R

  ## Score: n P_n l_beta = sum_i Z_i * (numerical derivative terms)
  num.1 <- exp.R * Lambda.R.exp - exp.L * Lambda.L.exp
  l1 <- Z * as.vector(num.1 / denum)
  Spsi <- colSums(l1)

  Spsi
}

#' Estimated information via the sPRES method (Section 2.3.4)
#'
#' Same Richardson-extrapolation scheme as [info_pres()], applied to the
#' closed-form profile score [fun_Spsi_2]. For each coordinate i the i-th
#' row of \eqn{\hat I_n} is
#' \deqn{(\hat I_n)_{i,\cdot} = -\{\tilde S(\tilde\beta^{(i)}_4) - 8
#' \tilde S(\tilde\beta^{(i)}_2) + 8 \tilde S(\tilde\beta^{(i)}_1)
#' - \tilde S(\tilde\beta^{(i)}_3)\} / (12 \epsilon).}
#'
#' @param lowerIC,upperIC Numeric vectors: interval endpoints; use
#'   \code{Inf} for right-censored subjects.
#' @param Z Covariate matrix (n x p).
#' @param b Numeric vector: the lasso estimate. By default only the nonzero
#'   subvector (the selected model) is used.
#' @param lambda Numeric vector: the hazard jumps of the adaptive lasso fit
#'   at \code{b} (used to start the EM updates).
#' @param h Positive scalar: the increment \eqn{\epsilon}.
#' @param tol,niter Convergence tolerance and iteration cap for the inner
#'   EM loop.
#' @param add Small positive constant for right-censored intervals.
#'
#' @return The symmetric \code{d x d} estimated information matrix
#'   \eqn{\hat I_{M,M}} (d = number of selected covariates).
#'
#' @seealso \code{\link{info_pres}}, \code{\link{PostSelectIC}}.
#' @references Zhang, J., Li, C., & Weng, H. (2025), Section 2.3.4.
#' @importFrom ALassoSurvIC fun_arglist
#' @export
info_spres <- function(lowerIC, upperIC, Z, b, lambda,
                       h = 1e-2, tol = 1e-6, niter = 1e5, add = 1e-5) {
  Mhat <- which(b != 0)
  b <- b[Mhat]
  d <- length(b)
  Z <- Z[, Mhat, drop = FALSE]
  trunc <- NULL
  normalize.X <- FALSE

  arglist <- ALassoSurvIC::fun_arglist(lowerIC, upperIC, Z, trunc,
                                       normalize.X, tol, niter)
  arglist$initial_lambda <- lambda

  information <- matrix(0, d, d)
  for (i in 1:d) {
    e_i <- rep(0, d)
    e_i[i] <- 1

    S1 <- fun_Spsi_2(arglist, b + h * e_i, add = add)
    S2 <- fun_Spsi_2(arglist, b - h * e_i, add = add)
    S3 <- fun_Spsi_2(arglist, b + 2 * h * e_i, add = add)
    S4 <- fun_Spsi_2(arglist, b - 2 * h * e_i, add = add)

    information[i, ] <- (S4 - 8 * S2 + 8 * S1 - S3) / (12 * h)
  }

  information <- -(information + t(information)) / 2
  information
}

#' Cluster-parallel sPRES information estimator
#'
#' Same estimator as [info_spres()], with the \eqn{4d} score evaluations
#' distributed over a \code{parallel} cluster. After use the cluster is
#' stopped (the original research code called \code{stopCluster} here).
#'
#' @param lowerIC,upperIC,Z,b,lambda,h,tol,niter,add As in [info_spres()].
#' @param cl A cluster object created by
#'   \code{\link[parallel]{makeCluster}}. If \code{NULL} (default), the
#'   computation runs serially and the cluster handling is skipped.
#' @param complete Logical. If \code{TRUE}, \code{b}, \code{Z} (and
#'   \code{lambda}) are assumed to be already restricted to the selected
#'   model; if \code{FALSE} (default) the nonzero subvector of \code{b} is
#'   used.
#'
#' @return The symmetric \code{d x d} estimated information matrix.
#'
#' @importFrom parallel clusterExport parRapply stopCluster
#' @export
info_spres_parallel <- function(lowerIC, upperIC, Z, b, lambda,
                                h = 1e-2, tol = 1e-6, niter = 1e5,
                                cl = NULL, complete = FALSE, add = 1e-5) {
  if (!complete) {
    Mhat <- which(b != 0)
    b <- b[Mhat]
    Z <- Z[, Mhat, drop = FALSE]
  }
  d <- length(b)
  trunc <- NULL
  normalize.X <- FALSE

  arglist <- ALassoSurvIC::fun_arglist(lowerIC, upperIC, Z, trunc,
                                       normalize.X, tol, niter)
  arglist$initial_lambda <- lambda

  information <- matrix(0, d, d)
  if (is.null(cl)) {
    for (i in 1:d) {
      e_i <- rep(0, d)
      e_i[i] <- 1

      S1 <- fun_Spsi_2(arglist, b + h * e_i, add = add)
      S2 <- fun_Spsi_2(arglist, b - h * e_i, add = add)
      S3 <- fun_Spsi_2(arglist, b + 2 * h * e_i, add = add)
      S4 <- fun_Spsi_2(arglist, b - 2 * h * e_i, add = add)

      information[i, ] <- (S4 - 8 * S2 + 8 * S1 - S3) / (12 * h)
    }
  } else {
    FUN <- function(x, arglist, add = add) {
      fun_Spsi_2(arglist, x, add = add)
    }

    X <- matrix(b, nrow = 4 * d, ncol = d, byrow = TRUE) +
      h * rbind(diag(1, d), diag(-1, d), diag(2, d), diag(-2, d))

    parallel::clusterExport(cl, "fun_Spsi_2")
    result <- parallel::parRapply(cl, X, FUN, arglist = arglist, add = add)
    parallel::stopCluster(cl)
    temp <- matrix(result, nrow = 4 * d, ncol = d, byrow = TRUE)
    for (i in 1:d) {
      information[i, ] <- (temp[3 * d + i, ] - 8 * temp[d + i, ] +
        8 * temp[i, ] - temp[2 * d + i, ]) / (12 * h)
    }
  }

  information <- -(information + t(information)) / 2
  information
}
