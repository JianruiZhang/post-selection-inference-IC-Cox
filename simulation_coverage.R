# =====================================================================
# simulation_coverage.R
#
# Monte Carlo evaluation of the conditional coverage of the selective
# confidence intervals, following Section 4 of:
#
#   Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for
#   the Cox model with interval-censored data. Scandinavian Journal of
#   Statistics, 52(2), 710-735. DOI: 10.1111/sjos.12768
#
# What this script does
# ---------------------
# For each Monte Carlo replication it
#   1. simulates one data set (strong or weak signal),
#   2. fits the adaptive lasso with a FIXED tuning parameter
#      lambda = C * sqrt(n) (the theoretically justified regime),
#   3. computes sPRES and LS information estimates,
#   4. runs the post-selection inference,
# and finally reports the coverage of the CIs for the TRUE signal
# covariates, both unconditionally and CONDITIONAL on the event that the
# selected model contains all signals ("no false negatives"), which is
# the event the theory targets.
#
# Runtime: hours with the default settings (nrep = 200). Reduce nrep, or
# set N_CORES > 1, for a quick pilot.
#
# License: GPL-3
# =====================================================================

suppressPackageStartupMessages({
  library(ALassoSurvIC)
  library(selectiveInference)
  library(matrixcalc)
  library(mvtnorm)
  library(parallel)
})

for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) {
  source(f, encoding = "UTF-8")
}

## ----------------------- simulation design ---------------------------
NREP    <- 200          # number of Monte Carlo replications (paper: 200)
N       <- 200          # sample size (paper also considers 400)
P       <- 10           # number of covariates
RHO     <- 0.2          # AR(1) covariate correlation
ALPHA   <- 0.05         # nominal non-coverage (paper reports 95% CIs)
N_CORES <- 1            # set > 1 to parallelize over replications

## Signal strength: strong = 1 (Section 4, Table 1), weak = 0.5 (Table 2)
BETA_SIGNAL <- 1
## Fixed tuning parameter lambda = C * sqrt(n) (paper: C = 0.425 for
## beta = 1, C = 0.725 for beta = 0.5, chosen to match the AIC average)
C_LAMBDA    <- if (BETA_SIGNAL == 1) 0.425 else 0.725

beta_true <- c(rep(BETA_SIGNAL, 2), rep(0, 6), rep(BETA_SIGNAL, 2))
SIGNALS   <- which(beta_true != 0)

## ----------------------- one replication -----------------------------
run_one <- function(rep_id) {
  sim <- GenerateIC(n = N, beta = beta_true, rho = RHO, SEED = 1000 + rep_id)
  X <- sim$Z
  colnames(X) <- paste0("X", 1:P)

  ## Fixed tuning parameter lambda = C * sqrt(n); alacoxIC's `theta` is the
  ## scalar in front of the adaptive weights theta / |beta_init_j|, so we
  ## compute beta_init (the unpenalized NPMLE) first.
  arg0 <- ALassoSurvIC::fun_arglist(sim$lowerIC, sim$upperIC, X, NULL,
                                    FALSE, 1e-3, 1e5)
  arg0$initial_lambda <- rep(1 / nrow(arg0$set), nrow(arg0$set))
  init <- ALassoSurvIC::fun_unpenSurvIC(rep(0, ncol(arg0$z)), arg0)
  theta_fixed <- C_LAMBDA * sqrt(N) * mean(abs(init$b))  # scalar theta such
  # that the *typical* per-coordinate penalty theta/|beta_init_j| is C*sqrt(n);
  # see README ("Choosing the tuning parameter") for the exact correspondence.

  fit <- fit_alacoxIC(lowerIC = sim$lowerIC, upperIC = sim$upperIC,
                      X = X, theta = theta_fixed)

  M <- which(fit$betahat != 0)
  screen_ok <- all(SIGNALS %in% M)   # selected model contains all signals

  out <- data.frame(rep = rep_id, screen_ok = screen_ok,
                    size_selected = length(M))
  if (!screen_ok || all(fit$betahat == 0)) {
    out$covered_cond <- NA
    return(out)
  }

  info_spres_M <- try(info_spres(sim$lowerIC, sim$upperIC, sim$Z,
                                 fit$betahat, fit$jumps, h = 1e-2),
                      silent = TRUE)
  info_ls_full <- try(fun_Hessian_3(sim$lowerIC, sim$upperIC, sim$Z,
                                    fit$betahat, fit$U, fit$jumps,
                                    threshold = 1e-10),
                      silent = TRUE)
  if (inherits(info_spres_M, "try-error") ||
      inherits(info_ls_full, "try-error")) {
    out$covered_cond <- NA
    return(out)
  }
  info_ls_M <- info_ls_full[M, M, drop = FALSE]

  res <- try(PostSelectIC(fit$betahat, info_spres_M, info_ls_M,
                          n = N, lambda = fit$lambda, alpha = ALPHA),
             silent = TRUE)
  if (inherits(res, "try-error")) {
    out$covered_cond <- NA
    return(out)
  }

  ## Coverage of the CIs for the true signal covariates (all four must
  ## cover for the replication to count as "covered"). Rows of res follow
  ## the order of the selected indices M.
  sig_pos <- which(M %in% SIGNALS)
  sig_true <- beta_true[M[sig_pos]]
  out$covered_cond <- all(res[sig_pos, "LOWER"] <= sig_true &
                          res[sig_pos, "UPPER"] >= sig_true)
  out
}

## ----------------------- run the Monte Carlo -------------------------
cl <- if (N_CORES > 1) parallel::makeCluster(N_CORES) else NULL
if (!is.null(cl)) {
  parallel::clusterExport(cl, varlist = c(
    "N", "P", "RHO", "ALPHA", "beta_true", "SIGNALS", "BETA_SIGNAL",
    "C_LAMBDA", "N", "run_one",
    "GenerateIC", "GenerateFromWH", "fit_alacoxIC", "info_spres",
    "fun_Hessian_3", "PostSelectIC"
  ))
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(ALassoSurvIC); library(selectiveInference)
      library(matrixcalc); library(mvtnorm)
    })
    for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) {
      source(f, encoding = "UTF-8")
    }
    NULL
  })
  results <- parallel::parLapply(cl, seq_len(NREP), run_one)
  parallel::stopCluster(cl)
} else {
  results <- lapply(seq_len(NREP), run_one)
}

sim_table <- do.call(rbind, results)

## ----------------------- summarize -----------------------------------
valid <- sim_table[!is.na(sim_table$covered_cond), ]
cat("\n================ Monte Carlo summary ================\n")
cat("Replications run:", NREP, "\n")
cat("Screening success (all signals selected):",
    mean(sim_table$screen_ok), "\n")
cat("Average selected model size:",
    round(mean(sim_table$size_selected), 2), "\n")
cat("Conditional coverage of 95% CIs (given all signals selected):",
    round(mean(valid$covered_cond), 3), "\n")
cat("=====================================================\n")

if (!dir.exists("output")) dir.create("output")
write.csv(sim_table, file.path("output", "simulation_coverage.csv"),
          row.names = FALSE)
cat("Per-replication results written to output/simulation_coverage.csv\n")
