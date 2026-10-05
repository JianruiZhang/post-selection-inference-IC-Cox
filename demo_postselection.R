# =====================================================================
# demo_postselection.R
#
# End-to-end demonstration of the post-selection inference pipeline:
#
#   1. Simulate one interval-censored data set (data-generation.R)
#   2. Fit the adaptive lasso (ALassoSurvIC::alacoxIC)   -> selection
#   3. Estimate the information matrices I-hat_{M,M}:
#        - sPRES (Section 2.3.4)  for the one-step estimator
#        - LS     (Section 2.3.2) for the pivot        ("sPRES+LS")
#   4. Run PostSelectIC() (Section 2.2)                  -> intervals
#
# Reference:
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for
#   the Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# Runtime: a few minutes on a typical laptop (dominated by the EM refits
# inside the information estimators).
#
# License: GPL-3
# =====================================================================

## Packages ------------------------------------------------------------
## install.packages(c("ALassoSurvIC", "selectiveInference", "matrixcalc", "mvtnorm"))
library(ALassoSurvIC)
library(selectiveInference)
library(matrixcalc)
library(mvtnorm)

## Source the pipeline -------------------------------------------------
## (If you use the package-style layout, source all files in R/.)
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) {
  source(f, encoding = "UTF-8")
}

## 1. Simulation setting of Section 4 (strong signal, beta = 1) --------
set.seed(2025)

p    <- 10
beta <- c(1, 1, rep(0, 6), 1, 1)   # signals at j = 1, 2, 9, 10
rho  <- 0.2                        # Sigma[i,j] = rho^|i-j|
n    <- 200

sim <- GenerateIC(n = n, beta = beta, rho = rho, SEED = 1)
cat("Interval-censored data generated:",
    sum(is.infinite(sim$upperIC)), "right-censored subjects out of", n, "\n\n")

## 2. Adaptive lasso fit (tuning parameter chosen by BIC) --------------
X   <- sim$Z
colnames(X) <- paste0("X", 1:p)
fit <- fit_alacoxIC(lowerIC = sim$lowerIC, upperIC = sim$upperIC, X = X)

cat("Adaptive lasso tuning parameter (BIC):", round(fit$theta, 3), "\n")
print(round(fit$betahat, 4))
cat("Selected model:", paste(which(fit$betahat != 0), collapse = ", "), "\n\n")

if (all(fit$betahat == 0)) {
  stop("The adaptive lasso selected no variables; nothing to infer about.")
}

## 3. Information matrices --------------------------------------------
## sPRES for the one-step estimator: returns I-hat_{M,M} (d x d), where
## d = number of selected covariates.
info_spres_M <- info_spres(
  lowerIC = sim$lowerIC, upperIC = sim$upperIC, Z = sim$Z,
  b = fit$betahat, lambda = fit$jumps, h = 1e-2
)

## LS for the pivot: returns the full p x p I-hat_n; subset to M.
info_ls_full <- fun_Hessian_3(
  lowerIC = sim$lowerIC, upperIC = sim$upperIC, Z = sim$Z,
  b = fit$betahat, U = fit$U, lambda = fit$jumps, threshold = 1e-10
)
M <- which(fit$betahat != 0)
info_ls_M <- info_ls_full[M, M, drop = FALSE]

## 4. Post-selection inference ----------------------------------------
res <- PostSelectIC(
  betahat   = fit$betahat,
  Hessian.1 = info_spres_M,   # one-step estimator  (sPRES)
  Hessian.2 = info_ls_M,      # pivot               (LS)
  n         = n,
  lambda    = fit$lambda,     # per-coordinate penalty theta/|beta_init|
  alpha     = 0.05
)

cat("\n================ Post-selection inference (sPRES+LS) ================\n")
print(round(res, 4))
cat("=====================================================================\n")
cat("betahat : adaptive lasso estimate\n")
cat("betabar : one-step estimator (equation 5 of the paper)\n")
cat("LOWER/UPPER : 95% selective confidence interval for the coefficient\n")
cat("p-value : two-sided selective p-value conditional on the selection event\n")

## For comparison: results using sPRES in BOTH steps -------------------
res_spres <- PostSelectIC(
  betahat = fit$betahat, Hessian.1 = info_spres_M, Hessian.2 = info_spres_M,
  n = n, lambda = fit$lambda, alpha = 0.05
)
cat("\n--- Variant: sPRES used in both steps ---\n")
print(round(res_spres, 4))

## Practical notes -----------------------------------------------------
## * CI columns LOWER/UPPER and 'betabar' estimate the PRESELECTION target
##   beta_{M,n} (the paper's tilde-theta_M); under correct selection they
##   target the true beta_M, but they remain valid even if some signal
##   covariates were missed (Section 2.2).
## * Naive intervals from ALassoSurvIC::unpencoxIC() on the selected model
##   ignore the selection event and undercover; that contrast is the point
##   of the method.
## * With normalize.X = FALSE (used by fit_alacoxIC) all quantities are on
##   the original covariate scale.
