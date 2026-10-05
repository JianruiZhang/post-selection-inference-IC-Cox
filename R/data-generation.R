# R/data-generation.R
#
# Simulate interval-censored data from the Cox proportional hazards model,
# as in Section 4 (Simulations) of:
#
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the
#   Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# Model
# -----
# Failure time T follows the Cox model with cumulative hazard
#
#     Lambda(t | Z) = (eta * t)^kappa * exp(beta' Z),
#
# i.e., a Weibull baseline cumulative hazard Lambda(t) = (eta * t)^kappa.
# Because R's `rweibull(shape, scale)` has cumulative hazard (t/scale)^shape,
# an equivalent sampler is
#
#     T ~ Weibull(shape = kappa, scale = exp(-beta' Z / kappa) / eta).
#
# Inspection process
# ------------------
# Each subject is examined at three random times V1 < V2 < V3 and only the
# bracketing interval of T is observed:
#
#     (0, V1]    interval-censored observation if  T <= V1
#     (V1, V2]   interval-censored observation if  V1 < T <= V2
#     (V2, V3]   interval-censored observation if  V2 < T <= V3
#     (V3, Inf)  right-censored observation       if  T > V3
#
# Covariates are multivariate normal with AR(1)-type covariance
# Sigma[i, j] = rho^|i - j|.
#
# License: GPL-3

#' Draw one failure time from the Weibull--Cox hazard
#'
#' Generates a single failure time T with cumulative hazard
#' \deqn{\Lambda(t \mid Z) = (\eta t)^{\kappa} \exp(\beta^\top Z)}
#' by inverting the equivalent \code{rweibull} parameterization
#' \code{Weibull(shape = kappa, scale = exp(-beta' Z / kappa) / eta)}.
#'
#' @param beta Numeric vector of regression coefficients (length p).
#' @param Z_i Numeric vector of covariates for this subject (length p).
#' @param kappa Positive scalar, the Weibull shape. Default 1.5 (paper).
#' @param eta Positive scalar, the Weibull cumulative-hazard scale:
#'   baseline cumulative hazard is \eqn{(\eta t)^{\kappa}}. Default 0.2.
#'
#' @return A single numeric failure time.
#'
#' @examples
#' set.seed(1)
#' GenerateFromWH(beta = c(1, 1, 0, 0), Z_i = c(0.5, -0.2, 1.0, 0.3))
#'
#' @export
GenerateFromWH <- function(beta, Z_i, kappa = 1.5, eta = 0.2) {
  if (length(beta) != length(Z_i)) {
    stop("'beta' and 'Z_i' must have the same length.")
  }
  if (kappa <= 0 || eta <= 0) {
    stop("'kappa' and 'eta' must be positive.")
  }
  rweibull(1, shape = kappa, scale = exp(sum(-beta * Z_i) / kappa) / eta)
}

#' Simulate interval-censored data from the Weibull--Cox model
#'
#' Generates \code{n} independent subjects under the model of
#' \code{\link{GenerateFromWH}} with AR(1)-type Gaussian covariates, then
#' applies the three-inspection-times censoring scheme described at the top
#' of this file. The random seed is set internally so that calls with the
#' same \code{SEED} reproduce the same data set.
#'
#' @param n Positive integer, the number of subjects.
#' @param beta Numeric vector of true regression coefficients (length p).
#' @param rho Correlation decay of the covariate covariance,
#'   \eqn{\Sigma_{ij} = \rho^{|i-j|}}. The simulations in the paper use 0.2.
#' @param SEED Integer seed passed to \code{\link{set.seed}}.
#' @param v1_range Length-2 vector: the uniform range of the first
#'   inspection time \eqn{U_1}. Default \code{c(3.2, 4.8)}.
#' @param gap_range Length-2 vector: the uniform range of each of the two
#'   inspection gaps \eqn{U_2 - U_1} and \eqn{U_3 - U_2}.
#'   Default \code{c(1.5, 2.5)}.
#' @param kappa,eta Passed to \code{\link{GenerateFromWH}}.
#'
#' @return A list with components:
#'   \item{lowerIC}{Numeric vector of length n: lower interval endpoints L.}
#'   \item{upperIC}{Numeric vector of length n: upper interval endpoints R
#'     (\code{Inf} for right-censored subjects).}
#'   \item{Z}{The n-by-p covariate matrix.}
#'
#' @importFrom mvtnorm rmvnorm
#' @export
GenerateIC <- function(n, beta, rho, SEED,
                       v1_range = c(3.2, 4.8),
                       gap_range = c(1.5, 2.5),
                       kappa = 1.5, eta = 0.2) {
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop("'n' must be a positive integer.")
  }
  if (rho < 0 || rho > 1) {
    stop("'rho' must lie in [0, 1].")
  }

  set.seed(SEED)
  p <- length(beta)

  ## AR(1)-type covariance: Sigma[i, j] = rho^|i - j|
  Sigma <- outer(seq_len(p), seq_len(p), function(i, j) rho^abs(i - j))

  ## Covariates and failure times
  Z <- mvtnorm::rmvnorm(n, sigma = Sigma)
  TT <- apply(Z, 1, function(Z_i) GenerateFromWH(beta, Z_i, kappa = kappa, eta = eta))

  ## Three random inspection times per subject
  V1 <- runif(n, min = v1_range[1], max = v1_range[2])
  V2 <- V1 + runif(n, min = gap_range[1], max = gap_range[2])
  V3 <- V2 + runif(n, min = gap_range[1], max = gap_range[2])

  ## Bracket each failure time with the inspection times:
  ## observed interval is (cuts[index], cuts[index + 1]].
  lowerIC <- upperIC <- numeric(n)
  for (i in seq_len(n)) {
    cuts <- c(0, V1[i], V2[i], V3[i], Inf)
    index <- findInterval(TT[i], cuts)
    lowerIC[i] <- cuts[index]
    upperIC[i] <- cuts[index + 1]
  }

  list(lowerIC = lowerIC, upperIC = upperIC, Z = Z)
}
