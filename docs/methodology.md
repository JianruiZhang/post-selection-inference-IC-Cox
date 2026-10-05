# Methodology guide

This page explains, end to end, what the code in this repository computes and
how each function maps to the methodology of the paper:

> Jianrui Zhang, Chenxi Li, and Haolei Weng (2025).
> **Post-selection inference for the Cox model with interval-censored data.**
> *Scandinavian Journal of Statistics* 52(2), 710–735.
> DOI: [10.1111/sjos.12768](https://doi.org/10.1111/sjos.12768)

Notation follows the paper. Small algorithmic details that are only visible in
the code (the `add` constant, the `threshold` filter, scaling conventions) are
flagged explicitly.

---

## Table of contents

1. [Model and observed data](#1-model-and-observed-data)
2. [Lasso estimation and the selection event](#2-lasso-estimation-and-the-selection-event)
3. [What goes wrong without selection adjustment](#3-what-goes-wrong-without-selection-adjustment)
4. [Post-selection target and the one-step estimator](#4-post-selection-target-and-the-one-step-estimator)
5. [The selective pivot: p-values and confidence intervals](#5-the-selective-pivot-p-values-and-confidence-intervals)
6. [Estimating the efficient information matrix](#6-estimating-the-efficient-information-matrix)
7. [Putting it together: the full pipeline](#7-putting-it-together-the-full-pipeline)
8. [Practical choices and numerical notes](#8-practical-choices-and-numerical-notes)
9. [Function-to-paper map](#9-function-to-paper-map)

---

## 1. Model and observed data

For subject $i$ let $T_i$ be the failure time and $X_i \in \mathbb{R}^p$ the
covariate vector. The failure time follows the Cox proportional hazards model

$$
\Lambda(t \mid X_i) \;=\; \Lambda(t)\,\exp(\beta^\top X_i), \tag{3}
$$

where $\Lambda$ is an unspecified baseline cumulative hazard and
$(\beta^*, \Lambda^*)$ denote the true values. The inspection process is a
random sequence of examination times $\vec U = (U_0, U_1, \dots, U_K, U_{K+1})$
with $U_0 = 0$, $U_{K+1} = \infty$, assumed **independent of $T$ given $X$**.
Only the bracketing interval of $T$ is observed:

$$
L_i = \max\{U_k : U_k < T_i\}, \qquad R_i = \min\{U_k : U_k \ge T_i\},
$$

so the observed data are $\{(\vec U_i, \vec\Delta_i, X_i)\}_{i=1}^n$, i.e.,
interval-censored data with possible right censoring ($R_i = \infty$).

The log likelihood of one observation is

$$
\ell(\beta, \Lambda) \;=\;
\log\Big[\exp\{-\Lambda(L_i)\,e^{\beta^\top X_i}\}
        -\exp\{-\Lambda(R_i)\,e^{\beta^\top X_i}\}\Big], \tag{4}
$$

and the full log likelihood is $n P_n \ell(\beta, \Lambda)$, where $P_n f$
denotes the empirical average.

**Code.** `GenerateIC()` and `GenerateFromWH()` in
[`R/data-generation.R`](../R/data-generation.R) simulate such data exactly as
in Section 4 of the paper: Weibull baseline cumulative hazard
$\Lambda(t) = (\eta t)^\kappa$ (with $\kappa = 1.5$, $\eta = 0.2$ or $0.5$
depending on the signal setting), $N_p(0, \Sigma)$ covariates with
$\Sigma_{ij} = \rho^{|i-j|}$, and three random inspection times per subject
$U_1 \sim \mathrm{Unif}(a, b)$, $U_2 = U_1 + \mathrm{Unif}(1.5, 2.5)$,
$U_3 = U_2 + \mathrm{Unif}(1.5, 2.5)$; a failure time beyond $U_3$ is
right-censored.

> **Numerical detail.** `rweibull(shape, scale)` has cumulative hazard
> $(t/\text{scale})^{\text{shape}}$, so sampling
> $T \sim \text{Weibull}(\kappa,\ \exp(-\beta^\top Z/\kappa)/\eta)$ is exactly
> equivalent to the hazard $\Lambda(t \mid Z) = (\eta t)^\kappa e^{\beta^\top Z}$.

---

## 2. Lasso estimation and the selection event

The lasso estimator solves

$$
(\hat\beta, \hat\Lambda) \;=\;
\operatorname*{argmax}_{(\beta, \Lambda)}\;
P_n \ell(\beta, \Lambda) \;-\; \frac{\lambda}{n}\,\|\beta\|_1 .
$$

Because the baseline hazard is infinite-dimensional, the optimization is
carried out by a **penalized EM algorithm** (Li, Pak and Todem, 2020): the M
step for $\beta$ is a shooting algorithm with an $\ell_1$ penalty, and the
baseline hazard jumps are updated in closed form on the *support set* (the
maximal intersections on which the NPMLE of $\Lambda$ can increase).

**Code.** The fitting is done by the CRAN package
[`ALassoSurvIC`](https://cran.r-project.org/package=ALassoSurvIC)
(`alacoxIC()`), which implements the adaptive lasso version: the penalty is

$$
\theta \sum_{j=1}^p \frac{|\beta_j|}{|\hat\beta^{\mathrm{init}}_j|},
\qquad
\hat\beta^{\mathrm{init}} = \text{unpenalized NPMLE},
$$

so the **effective per-coordinate penalty** entering the KKT conditions is

$$
\lambda_j \;=\; \frac{\theta}{|\hat\beta^{\mathrm{init}}_j|}. \tag{$\star$}
$$

`fit_alacoxIC()` in [`R/alacoxIC-wrapper.R`](../R/alacoxIC-wrapper.R) fits the
model with `normalize.X = FALSE` and assembles everything the inference
pipeline needs: the estimate `betahat = fit$b`, the per-coordinate penalties
`lambda` according to $(\star)$, the support set `U = fit$lambda.set[, 2]`, the
hazard jumps `jumps = fit$lambda`, and the sample size.

The selected model and the sign vector are

$$
\hat M = \{j : \hat\beta_j \neq 0\}, \qquad
\hat s_M = \big(\operatorname{sign}(\hat\beta_j)\big)_{j \in \hat M}.
$$

---

## 3. What goes wrong without selection adjustment

Naively, one would take the selected model $\hat M$ as given, refit
(unpenalized) on the selected covariates, and report likelihood-based standard
errors and Wald intervals. This ignores that $\hat M$ and $\hat s_M$ were
**chosen using the same data**. The sampling distribution of the refit
estimator *conditional on the selection event* is not the nominal one: Wald
intervals undercover for weak signals, and p-values are not uniform under the
null. The simulation studies in Sections 4 of the paper (and the comparison
against naive intervals) illustrate exactly this.

The framework used here is **conditional post-selection inference** (Lee et
al., 2016; Taylor and Tibshirani, 2018; Fithian et al., 2014): construct
intervals $C^M_j$ such that

$$
P\Big(\beta^M_j \in C^M_j \;\Big|\; \hat M = M,\ \hat s_M = s_M\Big)
\;\ge\; 1 - \alpha ,
\tag{2}
$$

at least asymptotically. Conditioning additionally on the signs is what makes
the selection event *polyhedral* (affine) in the target statistic, which is
the key computational trick — and it only costs power, never validity, since
$(2)$ implies marginal coverage by averaging over signs.

---

## 4. Post-selection target and the one-step estimator

The one-step estimator (equation (5) of the paper) is

$$
\tilde\beta_M \;=\; \hat\beta_M + \frac{\lambda}{n}\,
\hat I^{-1}_{M,n}\, s_M ,
\qquad
\hat I_{M,n} = \big(\hat I_n\big)_{M,M}, \tag{5}
$$

where $\hat I_n$ is a consistent estimator of $n \times$ the efficient
information matrix $\mathcal I$ evaluated at the estimate (Section 6 below),
and $s_M$ is the sign vector. Intuitively, the lasso estimate is shrunk along
the direction $s_M$ by exactly the amount of the penalty's KKT pull
$\frac{\lambda}{n}\mathcal I^{-1}_{M,M} s_M$; the one-step estimator undoes
that pull. It solves the estimating equation

$$
\frac{\partial}{\partial \beta_M}
P_n \ell\big((\beta_M, 0), \hat\Lambda(\beta_M)\big) = \frac{\lambda}{n} s_M ,
\tag{6}
$$

i.e., it is the profile-likelihood score of the selected model, re-centered at
the penalty. Because of the KKT conditions of the lasso, $\hat\beta_M$ itself
satisfies (6), so **no refitting is needed**: $\tilde\beta_M$ is computed
directly from $\hat\beta_M$ via (5).

Two remarks from the paper worth keeping in mind:

* The natural conditioning target is the *preselection* asymptotic mean
  $\tilde\theta_M = \theta^*_M + \mathcal I^{-1}_{M,M}\mathcal I_{M,-M}\theta^*_{-M}$
  (equation (10)); as long as $M$ contains all signal covariates (overselection
  included), $\tilde\theta_M = \theta^*_M$.
* Even when $M$ misses some signals, intervals for $\tilde\theta_M$ remain
  asymptotically valid for $\theta^M_n$, the maximizer of the population
  likelihood under the submodel (the standard "misspecified submodel" target),
  because $\tilde\theta_M = \theta^M_n + o(1)$ (Section S8 of the paper).

In the code, `PostSelectIC()` works with the $\sqrt n$-scaled quantities
$\tilde\theta_M = n^{1/2}\tilde\beta_M$ internally and reports results back on
the $\beta$ scale (`betabar` column).

---

## 5. The selective pivot: p-values and confidence intervals

The engine of the inference is a pivotal quantity that converges to a
**uniform distribution** under local alternatives, conditional on the
selection event. Writing the selection event as the affine constraint

$$
A\,\theta_M \le b, \qquad
A = -\operatorname{diag}(s_M), \qquad
b = -\big(n^{-1/2}\lambda\big)\operatorname{diag}(s_M)\,
    \hat I^{-1}_{M,M}\, s_M ,
$$

(for a scalar $\lambda$; with per-coordinate penalties $\lambda_j$ the vector
$b$ uses those), and for a direction $\gamma$ defining

$$
c = \hat I^{-1}_{M,M}\gamma\,\big(\gamma^\top \hat I^{-1}_{M,M}\gamma\big)^{-1},
\qquad
z = (I - c\gamma^\top)\,\tilde\theta_M ,
$$

the paper shows (equation (7), via Le Cam's third lemma and asymptotic
continuity of the empirical process) that

$$
F^{V_-,V_+}_{\gamma^\top \tilde\theta_M,\;
      \gamma^\top \hat I^{-1}_{M,M}\gamma}
\big(\gamma^\top \tilde\theta_M\big)
\;\Big|\; \{\hat M = M, \hat s_M = s_M\}
\;\;\overset{d}{\longrightarrow}\;\; \mathrm{Unif}(0, 1),
$$

where $F^{u,v}_{\mu,\sigma^2}$ is the CDF of $N(\mu, \sigma^2)$ truncated to
$[u, v]$ and

$$
V_- = \max_{j:\,(Ac)_j < 0} \frac{b_j - (Az)_j}{(Ac)_j},
\qquad
V_+ = \min_{j:\,(Ac)_j > 0} \frac{b_j - (Az)_j}{(Ac)_j}. \tag{8,9}
$$

A two-sided $1-\alpha$ selective confidence interval for $\gamma^\top
\tilde\theta_M$ is obtained by solving

$$
F^{V_-,V_+}_{\mathcal U,\;\gamma^\top \hat I^{-1}_{M,M}\gamma}
\big(\gamma^\top \tilde\theta_M\big) = \tfrac{\alpha}{2}
\quad\text{and}\quad
F^{V_-,V_+}_{\mathcal L,\;\gamma^\top \hat I^{-1}_{M,M}\gamma}
\big(\gamma^\top \tilde\theta_M\big) = 1 - \tfrac{\alpha}{2}
$$

for $(\mathcal L, \mathcal U)$ (monotonicity of $F$ in $\mu$ makes the
root-finding well posed). Setting $\gamma = e_j$ gives coordinate-wise
intervals; the two-sided p-value for $H_0: \tilde\theta_{M,j} = 0$ is obtained
from the same truncated-Gaussian CDF.

**Code.** All of the truncated-Gaussian machinery is delegated to
`selectiveInference::TG.interval()` and `selectiveInference::TG.pvalue()`
(the same battle-tested implementation used for the linear-model case).
`PostSelectIC()` in [`R/post-select-ic.R`](../R/post-select-ic.R)

1. validates the inputs and (unless `complete = TRUE`) restricts
   $\hat\beta$, the two information matrices and $\lambda$ to $\hat M$;
2. builds $A$ and $b$ from the sign vector;
3. forms the scaled one-step estimator
   $\tilde\theta_M = n^{1/2}\hat\beta_M + A b$;
4. for each $j$, calls `TG.*` with $\eta = e_j/\sqrt{n}$ and
   $\Sigma = n\hat I^{-1}_{M,M}$ (the second information estimate) and
   collects `LOWER`, `UPPER` and the two-sided p-value
   `2 * min(pv, 1 - pv)`.

The two information arguments exist because $\hat I_{M,M}$ is used **twice**
— once in the one-step estimator (5) and once in the pivot — and the
asymptotics only require *both* estimates to be consistent. This enables the
"method A + method B" configurations below.

> **Why the nuisance $\hat\Lambda_M$ can be ignored.** The selection event
> also involves the baseline hazard estimate. The paper shows (Section 3.1
> and S2.2–S2.5) that the $\hat\Lambda_M$-dependent part of the event is
> asymptotically independent of $\tilde\beta_M$, and the remaining part is
> exactly the affine constraint above — so discarding it changes nothing
> asymptotically while gaining power.

---

## 6. Estimating the efficient information matrix

The efficient information $\mathcal I$ for $\beta$ at $(\beta^*, \Lambda^*)$
has no closed form in this semiparametric model, and the *observed*
information of the profile likelihood is expensive and delicate to
differentiate numerically. The paper proposes and proves consistent three
practical estimators of $\hat I_n \approx n\mathcal I$; all three are
implemented here. Throughout, $(u_1, u_1'], \dots, (u_m, u_m']$ denote the
maximal intersections (support set) of $\hat\Lambda$, carried by
`fit$lambda.set` from `alacoxIC`.

### 6.1 Least squares approach — `fun_Hessian_3()` (Section 2.3.2)

Following Huang, Zhang and Hua (2012): among the linear space $\mathcal G_n$
spanned by the indicators $I(t \ge u_k)$, find the least squares direction
$\hat g_n$ matching the $\beta$-score, then estimate

$$
\hat I_n \;=\; n\Big(A_{11} - A_{12}A_{22}^{-}A_{21}\Big),
$$

with $A_{11} = P_n\{\ell_\beta{}^{\otimes 2}\}$,
$A_{12} = P_n\{\ell_\beta\,\dot\ell_\Lambda(\tilde g)^\top\}$,
$A_{22} = P_n\{\dot\ell_\Lambda(\tilde g)^{\otimes 2}\}$ and $A_{22}^{-}$ a
generalized inverse (`MASS::ginv`). This is a **Schur complement**: it removes
from the $\beta$-score variance the part explained by the least favorable
directions of the nuisance.

Implementation notes:

* The per-subject $\beta$-score ($l_1$) and the per-subject Fisher scores
  along $I(t \ge u_k)$ ($l_2$) are computed in closed form from
  $(\hat\beta, \hat\lambda, \text{support set})$.
* `threshold` (default `1e-10`) drops support points whose estimated jump is
  essentially zero — the adaptive lasso fit typically carries many such
  points, and keeping them makes $A_{22}$ needlessly large and ill-conditioned.
  Set `threshold = 0` to keep everything.
* The function returns the **full** $p \times p$ matrix $\hat I_n$; restrict
  it with `Ihat[M, M, drop = FALSE]` before passing to `PostSelectIC()`.
* In the paper's finite-sample experiments the LS estimate of the information
  tends to be **smaller** than the PRES/sPRES ones, which makes the resulting
  intervals conservative but stable — hence the recommended hybrid below.

### 6.2 PRES — `info_pres()` (Section 2.3.3)

Adapts Xu, Baines and Wang (2014) to this model. The key identity is that at
any working value $\tilde\beta$, the gradient of the *expected complete-data*
log likelihood of the EM algorithm, evaluated at
$(\tilde\beta, \hat\Lambda_{\tilde\beta})$, equals the score of the **profile
log likelihood**, $S(\tilde\beta) = n P_n \ell_\beta(\tilde\beta)$. The
function `fun_Spsi()` computes $S(\tilde\beta)$ by (a) running the EM hazard
updates to convergence at $\tilde\beta$ and (b) evaluating the closed-form
complete-data score. The information is then a fourth-order **Richardson
extrapolation** of $S$:

$$
(\hat I_n)_{i,\cdot} \;=\; -\frac{
  S(\hat\beta - 2\epsilon e_i) - 8\,S(\hat\beta - \epsilon e_i)
  + 8\,S(\hat\beta + \epsilon e_i) - S(\hat\beta + 2\epsilon e_i)}
  {12\,\epsilon}, \tag{PRES}
$$

evaluated for each selected coordinate $i$, then symmetrized. The result is
returned on the $d \times d$ selected block only (it subsets $\hat\beta$
internally).

### 6.3 sPRES — `info_spres()` (Section 2.3.4)

The paper's simplification of PRES. The profile score admits the closed form

$$
S(\tilde\beta) \;=\; n P_n\,\ell_\beta\big(\tilde\beta,\,
   \hat\Lambda_{\tilde\beta}\big),
$$

so the complete-data score evaluation can be replaced by the **observed-data
score** at the EM-updated hazard (`fun_Spsi_2()`); everything else,
including the Richardson extrapolation, is identical to PRES. sPRES is
cheaper and, in the simulations, slightly more stable than PRES at equal
validity.

### 6.4 Parallel variants — `info_pres_parallel()`, `info_spres_parallel()`

Each information estimate requires $4d$ score evaluations ($d$ = selected
model size). The `_parallel` variants distribute those over a
`parallel::makeCluster` cluster:

```r
cl <- parallel::makeCluster(4L)
I_spres <- info_spres_parallel(lowerIC, upperIC, Z, betahat, jumps, cl = cl)
```

(For the cluster path the score function is exported to the workers; the
cluster is stopped after use, as in the original research code. `cl = NULL`
runs serially.)

### 6.5 Which method, where?

Because the one-step estimator and the pivot may use two *different*
consistent estimates, the paper distinguishes:

| Configuration | one-step estimator (5) | pivot (7) | `Hessian.1` | `Hessian.2` |
|---|---|---|---|---|
| `sPRES+LS` (recommended) | sPRES | LS | `info_spres()` | `fun_Hessian_3()` block |
| `PRES+LS` | PRES | LS | `info_pres()` | `fun_Hessian_3()` block |
| `sPRES` | sPRES | sPRES | `info_spres()` | `info_spres()` |
| `PRES` | PRES | PRES | `info_pres()` | `info_pres()` |

In the simulations, `PRES+LS`/`sPRES+LS` are slightly conservative but
improve with $n$; plain `PRES`/`sPRES` are closer to nominal and already
behave well at $n = 200$. The profile-likelihood method of Section 2.3.1
(Murphy and van der Vaart, 2000) is theoretically clean but computationally
heavy and sensitive to the increment; it is **not** included in the
repository (it was not used in the paper's simulations either).

---

## 7. Putting it together: the full pipeline

```r
## 0. packages + the code in R/
library(ALassoSurvIC)
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) source(f)

## 1. data (or your own interval-censored data)
sim <- GenerateIC(n = 200, beta = c(1, 1, rep(0, 6), 1, 1), rho = 0.2, SEED = 1)

## 2. lasso selection + assembled penalty/scale inputs
X <- sim$Z; colnames(X) <- paste0("X", 1:10)
fit <- fit_alacoxIC(sim$lowerIC, sim$upperIC, X)     # theta by BIC, or fit$theta

## 3. two information estimates on the selected block
M <- which(fit$betahat != 0)
I1 <- info_spres(sim$lowerIC, sim$upperIC, sim$Z, fit$betahat, fit$jumps)
I2 <- fun_Hessian_3(sim$lowerIC, sim$upperIC, sim$Z, fit$betahat,
                    fit$U, fit$jumps)[M, M, drop = FALSE]

## 4. selective p-values and CIs (95% here)
res <- PostSelectIC(fit$betahat, I1, I2, n = fit$n, lambda = fit$lambda,
                    alpha = 0.05)
```

The runnable version of this is [`demo_postselection.R`](../demo_postselection.R);
the Monte Carlo coverage study of Section 4 is
[`simulation_coverage.R`](../simulation_coverage.R).

**Output columns.** `betahat` = $\hat\beta_M$; `betabar` = the one-step
estimator $\tilde\beta_M$; `LOWER`/`UPPER` = the selective $1-\alpha$
confidence interval; `p-value` = two-sided selective p-value for
$H_0: \tilde\theta_{M,j} = 0$. Rows are labeled by covariate name when `X`
has column names.

---

## 8. Practical choices and numerical notes

* **Tuning parameter.** With `theta` left missing, `alacoxIC` chooses it by
  BIC — reasonable in practice, though without theory support. The theory
  (and the paper's main tables) use a fixed $\lambda = C\sqrt n$; with the
  adaptive lasso, the scalar `theta` supplied to `alacoxIC` relates to the
  per-coordinate penalties through $(\star)$ above, and
  [`simulation_coverage.R`](../simulation_coverage.R) shows the exact
  correspondence used in the paper's regime. Note that with per-coordinate
  penalties, `PostSelectIC(lambda = fit$lambda)` is the correct call; the
  vector form is supported precisely for this reason.
* **Increment $\epsilon$ (`h`).** The simulations used
  $\epsilon \in \{10^{-2}, 10^{-5}, 10^{-7}\}$ and found the methods
  insensitive to the choice; the default `h = 1e-2` is the paper's main
  setting. Very small values can underflow the EM-based score, very large
  values bias the extrapolation.
* **Normalization.** Everything is fitted and inferred on the **original**
  covariate scale (`normalize.X = FALSE`). If you fit with
  `normalize.X = TRUE`, you must back-transform $\hat\beta$ and the
  information matrices before calling `PostSelectIC()`, otherwise the
  intervals are wrong (silently). `fit_alacoxIC()` handles this by always
  fitting un-normalized.
* **Right censoring.** Right-censored subjects have `upperIC = Inf`. A tiny
  constant `add = 1e-5` is used internally so that "the jump at
  $R_i + \text{add}$" is strictly inside the interval for numerical
  comparisons; this mirrors the original research code and does not affect
  the results.
* **Positive definiteness.** `PostSelectIC()` refuses non-PD information
  matrices (`matrixcalc::is.positive.definite`). Non-PD estimates can occur
  with very small selected models, extreme censoring, or unlucky LS
  Schur complements; the usual remedies are to use sPRES for the offending
  slot, to re-run with a different seed, or to increase $n$.
* **`lambda = Inf` entries.** If some $\hat\beta^{\mathrm{init}}_j = 0$, the
  per-coordinate penalty $(\star)$ is `Inf` for those coordinates. This is
  harmless as long as those coordinates are **not selected**
  ($\hat\beta_j = 0$), since only the selected block of $\lambda$ enters the
  computation; `PostSelectIC()` errors if a selected coordinate ever carries
  a non-finite penalty, which would indicate a degenerate fit.
* **Reproducibility.** `GenerateIC()` seeds internally via `SEED`, so
  simulation scripts are exactly reproducible.

---

## 9. Function-to-paper map

| Paper object | Equation / section | Code |
|---|---|---|
| Cox model with interval censoring | (3), Section 2.1 | `GenerateFromWH()`, `GenerateIC()` |
| Log likelihood | (4), Section 2.2 | (inside `fun_Spsi_2()`, `fun_Hessian_3()`) |
| Lasso / adaptive lasso estimator | Section 2.2 | `fit_alacoxIC()` → `ALassoSurvIC::alacoxIC()` |
| Selection event $\{\hat M = M, \hat s_M = s_M\}$ | Section 1–2.2 | `M = which(fit$b != 0)`; `A`, `b` inside `PostSelectIC()` |
| One-step estimator | (5)–(6) | `PostSelectIC()` (`thetabar`) |
| Selective pivot and intervals | (7)–(9) | `PostSelectIC()` → `selectiveInference::TG.*` |
| Efficient information, LS approach | Section 2.3.2 | `fun_Hessian_3()` |
| PRES | Section 2.3.3 | `info_pres()`, `info_pres_parallel()` |
| sPRES | Section 2.3.4 | `info_spres()`, `info_spres_parallel()` |
| Profile likelihood method | Section 2.3.1 | not included (see Section 6.5 above) |

---

## References

* Huang, J., Zhang, Y., & Hua, L. (2012). Consistent variance estimation in
  interval-censored data. In D.-G. Chen, J. Sun, & K. E. Peace (Eds.),
  *Interval-censored time-to-event data: Methods and applications* (pp.
  233–268). Chapman and Hall/CRC.
* Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016). Exact post-selection
  inference, with application to the lasso. *Annals of Statistics*, 44(3),
  907–927.
* Li, C., Pak, D., & Todem, D. (2020). Adaptive lasso for the Cox regression
  with interval censored and possibly left truncated data. *Statistical
  Methods in Medical Research*, 29(9), 2496–2512.
  (The preprint circulated as Li, Pak & Todem, 2019.)
* Murphy, S. A., & van der Vaart, A. W. (2000). On profile likelihood.
  *Annals of Statistics*, 28(3), 649–710.
* Taylor, J. E., & Tibshirani, R. J. (2018). Post-selection inference for
  $\ell_1$-penalized likelihood models. *Canadian Journal of Statistics*,
  46(1), 41–61.
* Tibshirani, R. J., Rinaldo, A., Tibshirani, R., & Wasserman, L. (2018).
  Uniform asymptotic inference and the bootstrap after model selection.
  *Annals of Statistics*, 46(6A), 2475–2502.
* Xu, C., Baines, P. D., & Wang, J.-L. (2014). Standard error estimation using
  the EM algorithm for the joint modeling of survival and longitudinal data.
  *Biostatistics*, 15(4), 731–744.
* Zhang, J., Li, C., & Weng, H. (2025). Post-selection inference for the Cox
  model with interval-censored data. *Scandinavian Journal of Statistics*,
  52(2), 710–735.
