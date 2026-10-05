# FUNCTION_MAP.md — function-by-function reference

Every function in `R/`, with signature, arguments, return value, and the
paper equation/section it implements. Math and background:
[docs/methodology.md](docs/methodology.md). Provenance with respect to the
original notebook: [docs/original-code-map.md](docs/original-code-map.md).

---

## R/data-generation.R

### `GenerateFromWH(beta, Z_i, kappa = 1.5, eta = 0.2)`

Draws one failure time from the Cox–Weibull hazard
$\Lambda(t \mid Z) = (\eta t)^{\kappa} \exp(\beta^\top Z)$, via the
equivalent `rweibull(shape = kappa, scale = exp(-beta'Z/kappa)/eta)`
parameterization.

* **Arguments**
  * `beta` — coefficient vector (length p).
  * `Z_i` — this subject's covariate vector (length p).
  * `kappa`, `eta` — Weibull shape and baseline cumulative-hazard scale
    ($\Lambda(t) = (\eta t)^\kappa$); paper defaults `1.5` / `0.2`.
* **Returns** a single numeric failure time.
* **Paper**: §2.1 model (3); §4 simulation design.

### `GenerateIC(n, beta, rho, SEED, v1_range = c(3.2, 4.8), gap_range = c(1.5, 2.5), kappa = 1.5, eta = 0.2)`

Simulates a full interval-censored data set: AR(1) Gaussian covariates
($\Sigma_{ij} = \rho^{|i-j|}$), Weibull–Cox failure times, three random
inspection times per subject, and the resulting censoring intervals
(`(0, V1]`, `(V1, V2]`, `(V2, V3]`, or right-censoring `(V3, Inf)`).

* **Arguments**
  * `n` — number of subjects.
  * `beta` — true coefficient vector (length p).
  * `rho` — covariate correlation decay (paper: 0.2).
  * `SEED` — seed for the internal `set.seed()` (makes runs reproducible).
  * `v1_range`, `gap_range` — inspection-time windows; defaults are the
    paper's strong-signal setting ($U_1 \sim \mathrm{Unif}(3.2, 4.8)$, gaps
    $\sim \mathrm{Unif}(1.5, 2.5)$).
* **Returns** list with `lowerIC` (n-vector L), `upperIC` (n-vector R, `Inf`
  for right-censored), `Z` (n × p covariate matrix).
* **Paper**: §2.1; §4.

---

## R/alacoxIC-wrapper.R

### `fit_alacoxIC(lowerIC, upperIC, X, theta, tol = 1e-3, niter = 1e5, cl = NULL, ...)`

Fits the adaptive lasso via `ALassoSurvIC::alacoxIC` with
`normalize.X = FALSE` and assembles all inputs required by the inference
pipeline.

* **Arguments**
  * `lowerIC`, `upperIC` — interval endpoints (`Inf` for right-censored).
  * `X` — covariate matrix (n × p); column names label the output rows.
  * `theta` — optional scalar tuning parameter; if missing, chosen by BIC
    inside `alacoxIC`.
  * `tol`, `niter` — EM convergence tolerance / iteration cap.
  * `cl` — optional `parallel` cluster, passed through.
  * `...` — further arguments to `alacoxIC`.
* **Returns** list with:
  * `betahat` — adaptive lasso estimate `fit$b` (p-vector, zeros =
    unselected);
  * `lambda` — **per-coordinate penalty vector** `theta / |unpen.b|`
    (entries `Inf` where `unpen.b` is 0; harmless unless that coordinate is
    selected);
  * `theta` — the scalar tuning parameter used;
  * `unpen.b` — unpenalized NPMLE initial estimate (`fit$unpen.b`);
  * `U` — support set right endpoints (`fit$lambda.set[, 2]`);
  * `jumps` — baseline hazard jumps (`fit$lambda`);
  * `n` — sample size; `fit` — the full `alacoxIC` object.
* **Paper**: §2.2 lasso estimation; penalty correspondence explained in
  `docs/methodology.md` §8.

---

## R/post-select-ic.R

### `PostSelectIC(betahat, Hessian.1, Hessian.2, n, lambda, alpha = 0.1, complete = FALSE)`

Selective two-sided p-values and `1 - alpha` confidence intervals for all
coordinates of the selected model, conditional on
$\{\hat M = M, \hat s_M = s_M\}$ (truncated-Gaussian pivot, equations
(7)–(9)).

* **Arguments**
  * `betahat` — lasso estimate $\hat\beta$ (p-vector; zeros identify
    unselected covariates).
  * `Hessian.1` — Fisher information $\hat I_{M,M}$ for the **one-step
    estimator** (negative Hessian of the profile log likelihood at
    $\hat\beta$). $d \times d$ if pre-restricted, otherwise p × p.
  * `Hessian.2` — Fisher information $\hat I_{M,M}$ for the **pivot**;
    may come from a different estimator than `Hessian.1` (hybrid
    configurations).
  * `n` — sample size.
  * `lambda` — tuning parameter: scalar, or **per-coordinate vector of
    length p** (`theta / |beta_init|` from `fit_alacoxIC()`).
  * `alpha` — level; intervals have coverage `1 - alpha` (use 0.05 for 95%).
  * `complete` — if `TRUE`, `betahat`, both Hessians and (a vector)
    `lambda` are already restricted to the selected model in a consistent
    order; if `FALSE` (default) the restriction is done internally.
* **Returns** matrix (one row per selected covariate) with columns
  `betahat`, `betabar` (one-step estimator, β scale), `LOWER`, `UPPER`
  (selective CI, β scale), `p-value` (two-sided).
* **Checks** — both Hessians positive definite; dimension agreement;
  `lambda` scalar or of length p.
* **Paper**: §2.2, equations (5)–(9).

---

## R/info-ls.R

### `fun_Hessian_3(lowerIC, upperIC, Z, b, U, lambda, threshold = 1e-10, add = 1e-5)`

Least squares information estimate $\hat I_n$ (Schur complement of empirical
score moments along the indicator directions of the support set) —
Section 2.3.2, following Huang, Zhang and Hua (2012). Original notebook
name kept.

* **Arguments**
  * `lowerIC`, `upperIC` — interval endpoints (`Inf` allowed).
  * `Z` — covariate matrix (n × p).
  * `b` — coefficient estimate at which to evaluate (typically
    `fit$betahat`; not subsetted internally).
  * `U` — support set right endpoints (`fit$U`).
  * `lambda` — corresponding hazard jumps (`fit$jumps`), same length as
    `U`.
  * `threshold` — drop support points with jump ≤ `threshold` (default
    `1e-10`; the adaptive lasso solution carries many numerically-zero
    jumps). Use `0` to keep all.
  * `add` — small constant for right-censored interval ends.
* **Returns** symmetric p × p matrix $\hat I_n \approx n\mathcal I$;
  restrict with `[M, M, drop = FALSE]` before use as a Hessian argument of
  `PostSelectIC()`.
* **Paper**: §2.3.2.

### `Mul(A, B)`

Row-wise outer-product accumulator
$\sum_i A[i,]\,B[i,]^\top$ used inside `fun_Hessian_3()`. Internal helper.

---

## R/info-pres.R

### `info_pres(lowerIC, upperIC, Z, b, lambda, h = 1e-2, tol = 1e-6, niter = 1e5, add = 1e-5)`

PRES information estimate (Section 2.3.3): Richardson-extrapolated
differentiation of the profile score `fun_Spsi()`, which is obtained by
re-running the EM hazard updates at each perturbed coefficient value.

* **Arguments**
  * `lowerIC`, `upperIC`, `Z` — data.
  * `b` — lasso estimate; the nonzero subvector (selected model) is used.
  * `lambda` — hazard jumps of the lasso fit (`fit$jumps`); starts the EM
    updates.
  * `h` — increment $\epsilon$ (paper: insensitive over
    $10^{-2}$–$10^{-7}$).
  * `tol`, `niter` — inner EM loop tolerance / cap.
  * `add` — right-censoring constant.
* **Returns** symmetric d × d matrix $\hat I_{M,M}$ (d = selected size).
* **Paper**: §2.3.3, equation (PRES) in `docs/methodology.md`.

### `info_pres_parallel(..., cl = NULL, complete = FALSE, ...)`

Same estimator with the 4d score evaluations distributed over a
`parallel` cluster `cl` (stopped after use, as in the original code);
`cl = NULL` runs serially. `complete = TRUE` treats `b`/`Z`/`lambda` as
already restricted to the selected model. Same return as `info_pres()`.

### `fun_Spsi(arglist, b, add = 1e-5)`

Profile score $S(\tilde\beta) = \partial Q / \partial\beta$ at
$(\tilde\beta, \hat\Lambda_{\tilde\beta})$, equal to
$nP_n\ell_\beta(\tilde\beta)$. Internal helper (documented in the source).

---

## R/info-spres.R

### `info_spres(lowerIC, upperIC, Z, b, lambda, h = 1e-2, tol = 1e-6, niter = 1e5, add = 1e-5)`

sPRES information estimate (Section 2.3.4): identical extrapolation scheme,
but the score is the closed-form observed-data profile score `fun_Spsi_2()`
at the EM-updated hazard — no complete-data evaluation.

* **Arguments** — as `info_pres()`.
* **Returns** symmetric d × d matrix $\hat I_{M,M}$.
* **Paper**: §2.3.4.

### `info_spres_parallel(..., cl = NULL, complete = FALSE, ...)`

Cluster-parallel variant of `info_spres()`; see `info_pres_parallel()`.

### `fun_Spsi_2(arglist, b, add = 1e-5)`

Closed-form profile score
$\tilde S(\tilde\beta) = nP_n\ell_\beta(\tilde\beta, \hat\Lambda_{\tilde\beta})$.
Internal helper.

---

## R/beta-blocks.R

Small utilities used by the (experimental) least squares path and available
for general use; the main pipeline needs none of them directly.

* `fun_sublr(u, l, r)` — n × m indicator matrix `A[i,k] = I(L_i < u_k <= R_i)`.
  Pure-R fallback for the compiled `ALassoSurvIC` helper.
* `fun_subless(u, lessthan)` — n × m indicator matrix
  `A[i,k] = I(u_k <= lessthan_i)`. Pure-R fallback.
* `fun_sgn(b)` — element-wise sign with exact zeros kept at 0.
* `fun_active(b)` — indices of nonzero entries of `b`.

---

## Suggested `library()` calls before sourcing

```r
library(ALassoSurvIC)      # alacoxIC, internal EM helpers (fun_ew, fun_updatelambda, fun_arglist)
library(mvtnorm)           # rmvnorm in GenerateIC
library(matrixcalc)        # is.positive.definite in PostSelectIC
library(MASS)              # ginv in fun_Hessian_3
library(selectiveInference) # TG.interval / TG.pvalue in PostSelectIC
```

(The R files call these packages through `::` wherever practical; loading
them up front keeps `demo_postselection.R` / `simulation_coverage.R`
self-contained.)
