# post-selection-inference-IC-Cox

[![DOI](https://img.shields.io/badge/DOI-10.1111%2Fsjos.12768-blue)](https://doi.org/10.1111/sjos.12768)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-4.x-276DC3)](https://www.r-project.org/)

Companion code for the paper

> **Jianrui Zhang, Chenxi Li, and Haolei Weng (2025).**
> Post-selection inference for the Cox model with interval-censored data.
> *Scandinavian Journal of Statistics* **52**(2), 710–735.
> DOI: [10.1111/sjos.12768](https://doi.org/10.1111/sjos.12768)

The paper develops p-values and confidence intervals for regression
coefficients that remain valid **after** data-driven variable selection by
lasso, in the Cox proportional hazards model with **interval-censored** data.
This repository contains the clean, documented R implementation refactored
from the authors' research notebook.

---

## What the method does

1. **Selects** a model $\hat M$ with the (adaptive) lasso for the
   interval-censored Cox model (penalized EM algorithm,
   [`ALassoSurvIC`](https://cran.r-project.org/package=ALassoSurvIC)).
2. Computes a **one-step estimator** $\tilde\beta_M$ that undoes the lasso
   shrinkage along the selected signs (equation (5) of the paper).
3. Conditions on the selection event $\{\hat M = M, \hat s_M = s_M\}$, which
   becomes an **affine constraint** on the scaled estimator, and builds
   truncated-Gaussian **p-values and confidence intervals** that are
   asymptotically valid under this conditioning (equations (7)–(9)).
4. Estimates the semiparametric **efficient information matrix** with three
   consistent methods — least squares (Section 2.3.2), **PRES**
   (Section 2.3.3) and **sPRES** (Section 2.3.4) — usable in either step,
   including the hybrid configurations ("sPRES+LS", "PRES+LS", ...) of the
   paper's simulations.

Unlike naive refit-and-infer, the intervals retain their nominal coverage
*conditional on the selected model* — the guarantee required in practice,
where the same data determined the model.

## Repository layout

```
.
├── R/                            # core functions (source these)
│   ├── data-generation.R         #   interval-censored data simulation (Section 2.1 / 4)
│   ├── alacoxIC-wrapper.R        #   lasso fit + assembled inputs for inference
│   ├── post-select-ic.R          #   PostSelectIC(): selective p-values & CIs (Section 2.2)
│   ├── info-ls.R                 #   least squares information (Section 2.3.2)
│   ├── info-pres.R               #   PRES information (Section 2.3.3)
│   ├── info-spres.R              #   sPRES information (Section 2.3.4)
│   └── beta-blocks.R             #   small helpers (fallback interval indicators)
├── demo_postselection.R          # runnable end-to-end demo (one data set)
├── simulation_coverage.R         # Monte Carlo coverage study (Section 4)
├── docs/
│   ├── methodology.md            # comprehensive method documentation
│   └── original-code-map.md      # mapping from the original "All Functions.Rmd"
├── FUNCTION_MAP.md               # function-by-function reference
├── CITATION.cff                  # citation metadata
├── LICENSE                       # GPL-3
└── README.md
```

## Requirements

* R ≥ 4.0
* CRAN packages: `ALassoSurvIC`, `selectiveInference`, `matrixcalc`,
  `mvtnorm`, `MASS`, `parallel` (base R).

```r
install.packages(c("ALassoSurvIC", "matrixcalc", "mvtnorm", "MASS"))
```

> **Note on `selectiveInference`.** The package has been archived from CRAN,
> but it remains fully functional and installable from the CRAN archive:
>
> ```r
> install.packages("remotes")
> remotes::install_version("selectiveInference", version = "1.2.5")
> ```

No installation step is needed for the code itself — clone or download the
repository and `source()` the files in `R/` (all cross-references use the
`::` namespace, so load order does not matter).

## Quick start

```r
## Load the pipeline
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) source(f)

## 1. Simulate one interval-censored data set (Section 4 setting, beta = 1)
sim <- GenerateIC(n = 200, beta = c(1, 1, rep(0, 6), 1, 1), rho = 0.2, SEED = 1)
X <- sim$Z; colnames(X) <- paste0("X", 1:10)

## 2. Adaptive lasso fit; assembles betahat, per-coordinate lambda, jumps, U
fit <- fit_alacoxIC(sim$lowerIC, sim$upperIC, X)          # theta by BIC

## 3. Information estimates on the selected block
M  <- which(fit$betahat != 0)
I1 <- info_spres(sim$lowerIC, sim$upperIC, sim$Z,         # sPRES (one-step)
                 fit$betahat, fit$jumps)
I2 <- fun_Hessian_3(sim$lowerIC, sim$upperIC, sim$Z,      # LS (pivot)
                    fit$betahat, fit$U, fit$jumps)[M, M, drop = FALSE]

## 4. Selective 95% CIs and two-sided p-values
res <- PostSelectIC(fit$betahat, I1, I2, n = fit$n,
                    lambda = fit$lambda, alpha = 0.05)
print(round(res, 4))
```

Columns of the result:

| column | meaning |
|---|---|
| `betahat` | lasso estimate $\hat\beta_M$ |
| `betabar` | one-step estimator $\tilde\beta_M$ (equation (5)) |
| `LOWER`, `UPPER` | selective $1-\alpha$ confidence interval |
| `p-value` | two-sided selective p-value, conditional on $\{\hat M = M, \hat s_M = s_M\}$ |

Run the full worked example with `Rscript demo_postselection.R` (a few
minutes on a laptop) and the Monte Carlo coverage study with
`Rscript simulation_coverage.R` (hours at the paper's 200 replications;
reduce `NREP` for a pilot).

## How it works (function ↔ paper map)

| Step | Function (file) | Paper |
|---|---|---|
| simulate data | `GenerateIC()`, `GenerateFromWH()` (`R/data-generation.R`) | §2.1, §4 |
| lasso selection + inputs | `fit_alacoxIC()` (`R/alacoxIC-wrapper.R`) | §2.2 |
| one-step estimator + selective inference | `PostSelectIC()` (`R/post-select-ic.R`) | §2.2, eq. (5)–(9) |
| information, least squares | `fun_Hessian_3()` (`R/info-ls.R`) | §2.3.2 |
| information, PRES | `info_pres()` / `info_pres_parallel()` (`R/info-pres.R`) | §2.3.3 |
| information, sPRES | `info_spres()` / `info_spres_parallel()` (`R/info-spres.R`) | §2.3.4 |

Configuration correspondence (`Hessian.1` = one-step, `Hessian.2` = pivot):

| paper's label | `Hessian.1` | `Hessian.2` |
|---|---|---|
| sPRES+LS (recommended) | `info_spres()` | `fun_Hessian_3()[M, M]` |
| PRES+LS | `info_pres()` | `fun_Hessian_3()[M, M]` |
| sPRES | `info_spres()` | `info_spres()` |
| PRES | `info_pres()` | `info_pres()` |

## Choosing the tuning parameter

* **BIC (default).** Call `fit_alacoxIC()` without `theta`;
  `alacoxIC()` picks the adaptive lasso tuning parameter by BIC. Practical,
  but without theory support.
* **Fixed penalty $\lambda = C\sqrt n$ (the theoretical regime).** The
  paper's theory assumes a fixed penalty of order $\sqrt n$
  ($C = 0.425$ for the strong-signal setting, $C = 0.725$ for the weak one).
  `alacoxIC` parameterizes the penalty as
  `theta / |beta_init_j|` per coordinate (`beta_init` = unpenalized NPMLE,
  `fit$unpen.b`), so `theta` corresponds to $C\sqrt n$ **times** the
  magnitude of the initial estimates; the exact per-coordinate penalties are
  always recovered by `fit_alacoxIC()` and must be passed to
  `PostSelectIC(lambda = fit$lambda)` as a **vector** — which is precisely
  what the wrapper produces. See `docs/methodology.md` (Section 8) for the
  full discussion.

## Documentation

* **[docs/methodology.md](docs/methodology.md)** — the complete walkthrough:
  model, lasso selection event, one-step estimator, selective pivot,
  all three information estimators, practical choices (increment, scale,
  censoring), and the full function-to-paper map.
* **[FUNCTION_MAP.md](FUNCTION_MAP.md)** — signatures, arguments, return
  values, and paper references for every exported function.
* **[docs/original-code-map.md](docs/original-code-map.md)** — what was
  kept/renamed/dropped relative to the original `All Functions.Rmd`, and why.

## Notes and caveats

* Fit and infer on the **same covariate scale**: the pipeline uses
  `normalize.X = FALSE` throughout (the wrapper enforces this). Mixing
  normalized and un-normalized quantities silently invalidates the intervals.
* `PostSelectIC(complete = TRUE)` skips the internal restriction to the
  selected model — use it only when `betahat`, both Hessians (and a vector
  `lambda`) are already restricted to the same subset, in the same order.
* `selectiveInference::TG.interval()` computes intervals on a bounded grid
  (`gridrange = c(-100, 100)` standard deviations by default); for extreme
  estimates widen it.
* The estimators require the information matrices to be positive definite;
  `PostSelectIC()` checks this and stops with an informative error. See
  `docs/methodology.md` §8 for remedies in small samples.

## Citing this work

If you use this code, please cite the paper:

```bibtex
@article{zhang2025postselection,
  author  = {Zhang, Jianrui and Li, Chenxi and Weng, Haolei},
  title   = {Post-selection inference for the {Cox} model with interval-censored data},
  journal = {Scandinavian Journal of Statistics},
  volume  = {52},
  number  = {2},
  pages   = {710--735},
  year    = {2025},
  doi     = {10.1111/sjos.12768}
}
```

(see also [CITATION.cff](CITATION.cff)).

## Acknowledgements

These repository files (the cleaned R modules, the demo and simulation
scripts, and the documentation under `docs/`) are created by
**GLM-5.3-Flash in AutoClaw**, refactored and documented from the authors'
original research notebook `All Functions.Rmd` — see
[docs/original-code-map.md](docs/original-code-map.md) for the full
provenance. The scientific method, the original code and the paper are the
work of the paper's authors.

## License

GPL-3 — the same license as the `ALassoSurvIC` package this work builds on;
see [LICENSE](LICENSE).
