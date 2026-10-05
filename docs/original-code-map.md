# Original-code map: from `All Functions.Rmd` to this repository

The original research notebook `All Functions.Rmd` (Jianrui Zhang, 2022) mixes
key methods with plotting scraps and superseded versions. This document states
exactly what was **kept**, what was **renamed**, what was **dropped**, and
what was **rewritten and why**, so every line of the new code is traceable to
the notebook.

---

## Summary table

| Original (in `All Functions.Rmd`) | Status here | New home |
|---|---|---|
| `GenerateFromWH` | kept, documented | `R/data-generation.R` |
| `GenerateIC` | kept, documented (two optional arguments added) | `R/data-generation.R` |
| `PostSelectIC` | **kept**, cleaned and documented | `R/post-select-ic.R` |
| `PostSelectIC_2` | dropped (superseded old version, per the author's note) | — |
| `fun_Spsi` + `fun_information_PRES` | kept as `fun_Spsi()` + `info_pres()` | `R/info-pres.R` |
| `fun_information_PRES_parallel` | kept as `info_pres_parallel()` | `R/info-pres.R` |
| `fun_Spsi_2` + `fun_information_PRES_2` | kept as `fun_Spsi_2()` + `info_spres()` | `R/info-spres.R` |
| `fun_information_PRES_parallel_2` | kept as `info_spres_parallel()` | `R/info-spres.R` |
| `Mul` + `fun_Hessian_3` ("Huang and Zhang") | kept as `Mul()` + `fun_Hessian_3()` | `R/info-ls.R` |
| `p.value.4 = p.value` and the two QQ-plot chunks | dropped (analysis leftovers, see below) | — |
| *(implicit)* adaptive lasso fit | new: `fit_alacoxIC()` wrapper | `R/alacoxIC-wrapper.R` |
| *(implicit)* `(0/1)`-interval helpers used by `ALassoSurvIC` | new: pure-R fallbacks `fun_sublr()`, `fun_subless()` | `R/beta-blocks.R` |

---

## What was kept, and how it was cleaned

### `GenerateFromWH` / `GenerateIC` → `R/data-generation.R`

Unchanged in behavior; the only additions are argument validation, roxygen
documentation, and two optional arguments (`v1_range`, `gap_range`) that
expose the inspection-time windows as parameters (with the paper's Section 4
values as defaults, so the default behavior is bit-for-bit the original
scheme given the same `SEED`). The covariate covariance is now built with
`outer()` instead of a double loop, and the failure-time loop with `apply()`.
Both original function names are kept as-is.

### `PostSelectIC` → `R/post-select-ic.R`

Same algorithm, same output. Cleanups:

* The commented-out p-value variants (one-sided `TG.pvalue` and the
  sign-flipped version) were removed; the active two-sided rule
  `2 * min(pv, 1 - pv)` is kept and documented.
* `drop = FALSE` added to all subsetting so that a single selected variable
  no longer collapses matrices to vectors.
* Row names are attached to the result when `betahat` carries covariate
  names.
* Calls are namespace-qualified (`selectiveInference::TG.interval`,
  `matrixcalc::is.positive.definite`) so the file cannot silently bind to a
  different function on the search path.
* The input contract is documented precisely (notably: `lambda` must be the
  **per-coordinate** penalty vector `theta / |beta_init|` of the adaptive
  lasso, not the scalar `theta`).

### PRES block (`fun_Spsi`, `fun_information_PRES`, `..._parallel`) → `R/info-pres.R`

* `fun_Spsi` is byte-for-byte the same computation, re-indented; the `print`
  progress statements of the notebook version were already only in the
  wrapper, and they are removed.
* `fun_information_PRES` is renamed `info_pres()`: same Richardson
  extrapolation `(S4 - 8 S2 + 8 S1 - S3) / (12 h)` and symmetrization.
* `fun_information_PRES_parallel` is renamed `info_pres_parallel()`,
  including the notebook's original (quirky but correct) `stopCluster(cl)`
  call after `parRapply` — documented in the roxygen text so the behavior is
  not a surprise. The block-row layout of the `X` design matrix is explained
  in comments.

### sPRES block (`fun_Spsi_2`, `fun_information_PRES_2`, `..._parallel_2`) → `R/info-spres.R`

Same treatment as the PRES block: `fun_Spsi_2` kept (it is the closed-form
observed-data profile score, Section 2.3.4), wrappers renamed
`info_spres()` / `info_spres_parallel()`. The commented-out dead code inside
`fun_Spsi_2` (the alternative `l1` loop) was removed.

### LS block (`Mul`, `fun_Hessian_3`) → `R/info-ls.R`

Both functions kept with their original names (this is the method the paper
calls "least squares", Section 2.3.2, following Huang, Zhang and Hua, 2012;
the notebook heading was "Huang and Zhang"). Documentation added; the
commented-out NA-handling fallback inside `fun_Hessian_3` was dropped. The
`threshold` filter (drop support points with jumps `<= threshold`) is kept
and now explained.

### New, not in the notebook

* **`R/alacoxIC-wrapper.R` (`fit_alacoxIC`).** In the notebook the lasso fit
  happened outside the kept chunks. The wrapper makes the pipeline
  reproducible and — crucially — assembles the **per-coordinate penalty**
  `lambda = theta / |unpen.b|` that `PostSelectIC()` requires. The original
  scripts computed this by hand; getting it wrong silently invalidates the
  intervals, which is why it is now centralized.
* **`R/beta-blocks.R`.** Pure-R fallbacks for the compiled helpers
  `fun_sublr`/`fun_subless` plus sign/active-set utilities, kept separate
  from the main pipeline because the main pipeline only needs them through
  `ALassoSurvIC` itself.

---

## What was dropped, and why

* **`PostSelectIC_2`.** The author's own annotation marks it as the old
  version ("could be neglected"). It differs from `PostSelectIC` in that it
  indexes results by the full `betahat` (rows for unselected variables stay
  zero) and uses a single-Hessian interface with `Hessian`/`Hessian_2`
  naming; all of that is superseded by `PostSelectIC(complete = ...)`.
* **`p.value.4 = p.value` and the two plotting chunks.** These are analysis
  *leftovers*, not methods: they re-plot precomputed p-value vectors
  (`p.value.1` ... `p.value.4`, produced by scripts not included in the
  notebook) into QQ plots and an EPS file (`n200b1e1e-5fixed.eps`). The
  variables they reference (`p.value.*`, `pointer`) do not exist in the
  notebook, so the chunks were never self-contained. Their purpose —
  comparing the four information-estimation configurations by QQ plots of
  selective p-values — is documented in
  [`methodology.md`](methodology.md) (the configuration table), and the
  pipeline to regenerate such p-values is `demo_postselection.R` /
  `simulation_coverage.R`.
* **Commented-out code and `print("S1 done")` progress statements** inside
  the preserved functions, as noted above.

---

## Behavior-preservation notes

* Every numerical formula (likelihood, EM updates, Richardson coefficients,
  Schur complement, truncated-Gaussian calls) is **identical** to the
  notebook; only names, comments, guards and formatting changed.
* Function signature changes are limited to: added trailing arguments with
  defaults (`v1_range`, `gap_range`), and the renaming table above.
  Call sites written against the old names keep working by prepending
  `source("R/data-generation.R")` etc., since the primary names
  (`GenerateFromWH`, `GenerateIC`, `PostSelectIC`, `fun_Spsi`, `fun_Spsi_2`,
  `fun_Hessian_3`, `Mul`) are unchanged.

---

*This refactoring and the accompanying documentation were created by
GLM-5.3-Flash in AutoClaw. All scientific content originates from the
authors' notebook and the published paper.*
