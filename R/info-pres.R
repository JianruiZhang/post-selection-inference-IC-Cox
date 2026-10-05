# R/info-pres.R
#
# PRES (penalized profile score) information estimator, Section 2.3.3 of:
#
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the
#   Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# The method adapts Xu, Baines and Wang (2014) to the interval-censored Cox
# model. At a working coefficient value b~, the EM algorithm of
# ALassoSurvIC is re-run to convergence, which yields the maximizer
# Lambda(b~) of the expected complete-data log likelihood Q. The gradient of
# Q at (b~, Lambda(b~)) equals the score of the profile log likelihood
# evaluated at b~. Differentiating this score by Richardson extrapolation
# estimates -P_n l_{beta beta}(betahat), used as I-hat_n for n * I.
#
# License: GPL-3

#' Profile score S(b~) of the expected complete-data log likelihood
#'
#' Evaluates
#' \deqn{S(\tilde\beta) = \left.\frac{\partial Q(\beta, \Lambda \mid
#' \tilde\beta, \hat\Lambda_{\tilde\beta})}{\partial \beta}\right|_{\beta =
#' \tilde\beta, \Lambda = \hat\Lambda_{\tilde\beta}},}
#' where \eqn{\hat\Lambda_{\tilde\beta}} is obtained by iterating the EM
#' update of \code{ALassoSurvIC} (via \code{fun_ew} and
#' \code{fun_updatelambda}) until the hazard jumps converge. By standard
#' calculus, \eqn{S(\tilde\beta)} equals the score of the profile log
#' likelihood, \eqn{n P_n l_\beta(\tilde\beta)}.
#'
#' @param arglist A list as produced by
#'   \code{\link[ALassoSurvIC]{fun_arglist}}, additionally carrying
#'   \code{initial_lambda} (the starting hazard jumps).
#' @param b Numeric vector: the working coefficient value \eqn{\tilde\beta}.
#' @param add Small positive constant: for right-censored subjects
#'   (\code{R = Inf}) the effective interval end is \code{R + add}.
#'
#' @return Numeric vector of length \code{length(b)}: the profile score.
#'
#' @references
#' Xu, C., Baines, P. D., & Wang, J.-L. (2014). Standard error estimation
#' using the EM algorithm for the joint modeling of survival and
#' longitudinal data. \emph{Biostatistics}, 15, 731-744.
#'
#' @keywords internal
#' @noRd
fun_Spsi <- function(arglist, b, add = 1e-5) {
  n <- arglist$n
  Z <- arglist$z
  L <- arglist$l
  R <- arglist$r
  U <- arglist$set[, 2]
  m <- length(U)
  d <- length(b)

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

  ## exp(Z_i' b) and Lambda_i(R + add)
  R_ast <- L * (R == Inf)
  R_ast[which(R < Inf)] <- R[which(R < Inf)]
  exp_Zb <- exp(Z %*% b)

  ## Expected complete-data contributions EW[i, k], zeroed outside
  ## the support of subject i's interval
  nom <- rep(0, n)
  for (i in 1:n) {
    nom[i] <- 1 - exp(-exp_Zb[i] * sum(lambda[which(U <= (R[i] + add) & U > L[i])]))
  }
  EW <- (exp_Zb / nom) %*% lambda
  for (k in 1:m) {
    ind <- which(L >= U[k])
    EW[ind, k] <- rep(0, length(ind))
  }
  for (k in 1:m) {
    for (i in 1:n) {
      if (U[k] <= L[i] || U[k] > (R[i] + add)) {
        EW[i, k] <- 0
      }
      if (L[i] < U[k] && U[k] <= R[i] && is.infinite(R[i])) {
        EW[i, k] <- 0
      }
    }
  }

  ## Score: sum over support points of EW[i,k] * (Z_i - weighted mean)
  Spsi <- rep(0, d)
  for (k in 1:m) {
    weight <- exp_Zb * (R_ast + add >= U[k])
    temp <- (t(weight) %*% Z) / sum(weight)
    for (i in 1:n) {
      Spsi <- Spsi + (U[k] <= (R_ast[i] + add)) * EW[i, k] * (Z[i, ] - temp)
    }
  }

  Spsi
}

#' Estimated information via the PRES method (Section 2.3.3)
#'
#' Builds \eqn{\hat I_{M,M}} by numerically differentiating the profile
#' score [fun_Spsi] with a fourth-order Richardson extrapolation. For each
#' coordinate \eqn{i}, the i-th row of \eqn{\hat I_n} is
#' \deqn{(\hat I_n)_{i,\cdot} = -\{S(\tilde\beta^{(i)}_4) - 8 S(\tilde\beta^{(i)}_2)
#' + 8 S(\tilde\beta^{(i)}_1) - S(\tilde\beta^{(i)}_3)\} / (12 \epsilon),}
#' with \eqn{\tilde\beta^{(i)}_1 = \hat\beta + \epsilon e_i},
#' \eqn{\tilde\beta^{(i)}_2 = \hat\beta - \epsilon e_i},
#' \eqn{\tilde\beta^{(i)}_3 = \hat\beta + 2 \epsilon e_i}, and
#' \eqn{\tilde\beta^{(i)}_4 = \hat\beta - 2 \epsilon e_i}.
#'
#' @param lowerIC,upperIC Numeric vectors: interval endpoints; use
#'   \code{Inf} for right-censored subjects.
#' @param Z Covariate matrix (n x p).
#' @param b Numeric vector: the lasso estimate. By default only the nonzero
#'   subvector (the selected model) is used.
#' @param lambda Numeric vector: the hazard jumps of the adaptive lasso fit
#'   at \code{b} (used to start the EM updates).
#' @param h Positive scalar: the increment \eqn{\epsilon}. The paper reports
#'   results for \eqn{\epsilon = 10^{-2}, 10^{-5}, 10^{-7}} and finds the
#'   method insensitive to this choice.
#' @param tol,niter Convergence tolerance and iteration cap for the inner
#'   EM loop.
#' @param add Small positive constant for right-censored intervals.
#'
#' @return The symmetric \code{d x d} estimated information matrix
#'   \eqn{\hat I_{M,M}} (d = number of selected covariates).
#'
#' @seealso \code{\link{info_pres_parallel}} for a cluster-parallel version,
#'   \code{\link{PostSelectIC}} for its use in post-selection inference.
#' @references Zhang, J., Li, C., & Weng, H. (2025), Section 2.3.3.
#' @importFrom ALassoSurvIC fun_arglist
#' @export
info_pres <- function(lowerIC, upperIC, Z, b, lambda,
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

    S1 <- fun_Spsi(arglist, b + h * e_i, add = add)
    S2 <- fun_Spsi(arglist, b - h * e_i, add = add)
    S3 <- fun_Spsi(arglist, b + 2 * h * e_i, add = add)
    S4 <- fun_Spsi(arglist, b - 2 * h * e_i, add = add)

    information[i, ] <- (S4 - 8 * S2 + 8 * S1 - S3) / (12 * h)
  }

  information <- -(information + t(information)) / 2
  information
}

#' Cluster-parallel PRES information estimator
#'
#' Same estimator as [info_pres()], with the \eqn{4d} score evaluations
#' distributed over a \code{parallel} cluster. After use the cluster is
#' stopped (the original research code called \code{stopCluster} here).
#'
#' @param lowerIC,upperIC,Z,b,lambda,h,tol,niter,add As in [info_pres()].
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
info_pres_parallel <- function(lowerIC, upperIC, Z, b, lambda,
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

      S1 <- fun_Spsi(arglist, b + h * e_i, add = add)
      S2 <- fun_Spsi(arglist, b - h * e_i, add = add)
      S3 <- fun_Spsi(arglist, b + 2 * h * e_i, add = add)
      S4 <- fun_Spsi(arglist, b - 2 * h * e_i, add = add)

      information[i, ] <- (S4 - 8 * S2 + 8 * S1 - S3) / (12 * h)
    }
  } else {
    FUN <- function(x, arglist, add = add) {
      fun_Spsi(arglist, x, add = add)
    }

    ## Rows 1:d          -> b + h e_i
    ## Rows (d+1):(2d)   -> b - h e_i
    ## Rows (2d+1):(3d)  -> b + 2h e_i
    ## Rows (3d+1):(4d)  -> b - 2h e_i
    X <- matrix(b, nrow = 4 * d, ncol = d, byrow = TRUE) +
      h * rbind(diag(1, d), diag(-1, d), diag(2, d), diag(-2, d))

    parallel::clusterExport(cl, "fun_Spsi")
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
