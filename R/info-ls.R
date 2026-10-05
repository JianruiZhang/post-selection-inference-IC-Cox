# R/info-ls.R
#
# Least squares (LS) information estimator, Section 2.3.2 of:
#
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the
#   Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# This is `fun_Hessian_3` in the original research notebook ("Huang and
# Zhang"). It follows Huang, Zhang and Hua (2012): the efficient information
# is estimated as the Schur complement
#
#     I-hat_n = n * (A11 - A12 A22^- A21),
#
# where the blocks are empirical second moments of the beta-score and of
# the Fisher scores of one-dimensional parametric submodels along the
# directions I(t >= u_k), u_k running over the support set (maximal
# intersections) of the estimated baseline cumulative hazard.
#
# License: GPL-3

#' Sum of outer products of rows
#'
#' Computes \code{C = sum_i A[i, ] t(B[i, ])}, i.e.,
#' \eqn{C = \sum_{i=1}^n a_i b_i^\top} where \eqn{a_i} and \eqn{b_i} are the
#' i-th rows of \code{A} and \code{B}. Used to accumulate the empirical
#' moment blocks A11, A12 and A22 row by row.
#'
#' @param A Numeric matrix (n x p).
#' @param B Numeric matrix (n x q).
#'
#' @return The \code{p x q} matrix \eqn{\sum_i a_i b_i^\top}.
#'
#' @keywords internal
#' @noRd
Mul <- function(A, B) {
  n <- nrow(A)
  C <- matrix(0, nrow = ncol(A), ncol = ncol(B))
  for (i in 1:n) {
    C <- C + A[i, ] %*% t(B[i, ])
  }
  C
}

#' Estimated information via the least squares approach (Section 2.3.2)
#'
#' Computes
#' \deqn{\hat I_n = n (A_{11} - A_{12} A_{22}^{-} A_{21}),}
#' \deqn{A_{11} = P_n\{l_\beta(\hat\beta, \hat\Lambda)^{\otimes 2}\},}
#' \deqn{A_{12} = P_n\{l_\beta(\hat\beta, \hat\Lambda)\,
#' \dot l_\Lambda(\hat\beta, \hat\Lambda)(\tilde g)^\top\},}
#' \deqn{A_{22} = P_n\{\dot l_\Lambda(\hat\beta, \hat\Lambda)(\tilde g)
#' ^{\otimes 2}\},}
#' where \eqn{\tilde g = (I(t \ge u_1), \dots, I(t \ge u_m))^\top} indexes
#' the jumps of \eqn{\hat\Lambda} over the support set, and \eqn{A_{22}^{-}}
#' is a generalized inverse (\code{MASS::ginv}). The result is symmetrized
#' before being returned.
#'
#' @param lowerIC,upperIC Numeric vectors: interval endpoints; use
#'   \code{Inf} for right-censored subjects.
#' @param Z Covariate matrix (n x p).
#' @param b Numeric vector: the coefficient estimate at which the
#'   information is evaluated (typically the lasso estimate; entries may be
#'   zero, the function does not subset them).
#' @param U Numeric vector: the right endpoints \eqn{u_1, \dots, u_m} of the
#'   support set (maximal intersections) of the estimated baseline hazard,
#'   i.e., \code{fit$lambda.set[, 2]} from \code{\link[ALassoSurvIC]{alacoxIC}}.
#' @param lambda Numeric vector: the corresponding hazard jumps
#'   \code{fit$lambda}; must have the same length as \code{U}.
#' @param threshold Nonnegative scalar. Support points whose jump is at
#'   most \code{threshold} are discarded (when \code{threshold > 0}).
#'   This removes the numerically near-zero jumps of the adaptive lasso
#'   solution; use 0 to keep every support point.
#' @param add Small positive constant: for right-censored subjects
#'   (\code{R = Inf}) the effective interval end is \code{R + add}.
#'
#' @return The symmetric \eqn{p \times p} matrix \eqn{\hat I_n}
#'   (an estimate of \eqn{n} times the efficient information \eqn{I}).
#'   Restrict to the selected model with \code{Ihat[M, M, drop = FALSE]}
#'   before passing it to [PostSelectIC()].
#'
#' @references
#' Huang, J., Zhang, Y., & Hua, L. (2012). Consistent variance estimation in
#' interval-censored data. In D.-G. Chen, J. Sun, & K. E. Peace (Eds.),
#' \emph{Interval-censored time-to-event data: Methods and applications}
#' (pp. 233-268). Chapman and Hall/CRC.
#'
#' Zhang, J., Li, C., & Weng, H. (2025), Section 2.3.2.
#'
#' @importFrom MASS ginv
#' @export
fun_Hessian_3 <- function(lowerIC, upperIC, Z, b, U, lambda,
                          threshold = 1e-10, add = 1e-5) {
  ## Optionally drop support points with negligible hazard jumps
  if (threshold > 0) {
    ind <- which(lambda > threshold)
    lambda <- lambda[ind]
    U <- U[ind]
  }

  n <- length(lowerIC)
  d <- length(b)
  L <- lowerIC
  R <- upperIC
  m <- length(lambda)

  exp_bt_Z <- exp(Z %*% b)

  ## Baseline cumulative hazard at L and at R + add, scaled by exp(Z b)
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

  ## l1: per-subject beta-score l_beta (n x d)
  num.1 <- exp.R * Lambda.R.exp - exp.L * Lambda.L.exp
  l1 <- Z * as.vector(num.1 / denum)

  ## l2: per-subject Fisher scores along the directions I(t >= u_k) (n x m)
  l2 <- num.2 <- matrix(0, nrow = n, ncol = m)
  for (j in 1:m) {
    num.2[, j] <- exp_bt_Z * (exp.R * ((R + add) >= U[j]) - exp.L * (L >= U[j]))
    l2[, j] <- num.2[, j] / denum
  }

  ## Moment blocks and the Schur complement
  A11 <- Mul(l1, l1)
  A12 <- Mul(l1, l2)
  A22 <- Mul(l2, l2)

  Hessian <- A11 - A12 %*% MASS::ginv(A22) %*% t(A12)
  Hessian <- (Hessian + t(Hessian)) / 2
  Hessian
}
