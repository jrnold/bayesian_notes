
# Shrinkage and Regularization {#shrinkage-regularization}

*Shrinkage estimation* deliberately introduces biases into the model to improve overall performance, often at the cost of individual estimates [@EfronHastie2016a, p. 91].

This is opposed to MLE, which produces unbiased estimates (asymptotically, given certain regularity conditions). Likewise, the Bayesian estimates with non- or weakly-informative priors will produce estimates similar to the MLE.
With shrinkage, the priors are used to produce estimates *different* than the MLE case.

*Regularization* describes any method that reduces variability in high dimensional estimation or prediction problems [@EfronHastie2016a].

## Normal Linear Regression Model

Consider the single output linear Gaussian regression model with several input variables, given by
$$
\begin{aligned}[t]
y_i \sim \dnorm(\vec{x}_i' \vec{\beta}, \sigma^2)
\end{aligned}
$$
where $\vec{x}$ is a $k$-vector of predictors, and $\vec{\beta}$ are the coefficients.

What priors do we put on $\beta$? 

- **Improproper priors:** $\beta_k \propto 1$ This produces the equivalent of MLE estimates.
- **Non-informative priors:** These are priors which have such wide variance that they have little influence on the posterior, e.g. $\beta_k \sim \dnorm(0, 1e6)$. The primary reason for these (as opposed to simply using an improper prior) is that some MCMC methods, e.g. Gibbs sampling as used in JAGS or BUGS, require proper prior distributions for all parameters.

**Shrinkage priors** have a couple characteristics

- they push $\beta_k \to 0$
- while in the other cases, the scale of the prior on $\beta$ is fixed, in shrinkage priors there is often a hyperprior on it. E.g. $\beta_k \sim \dnorm(0, \tau)$, where $\tau$ is also a parameter to be estimated.

## Penalized Regression

Penalized regressions are regressions of the form:
$$
\hat{\beta}_{penalized} = \argmin_{\beta} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + f(\beta)
$$
where $f$ is some sort of penalty function on $\beta$ that penalizes larger (in magnitude) values of $\beta$.

Two common forms

- Ridge: uses an $\ell_2$ penalty: $\vec{beta}^2$
- Lasso: uses an $\ell_1$ penalty: $|\vec{\beta}|$

### Ridge Regression

Ridge regression uses the following penalty [@HoerlKennard1970a]:
$$
\hat{\beta}_{\text{ridge}} = \argmin_{\beta} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + \lambda \sum_{k} \beta_k^2
$$
This penalty produces smaller in magnitude coefficients, $|\hat{\beta}_{ridge}| < |\hat{\beta}_{OLS}|$.
However, this "bias" in the coefficients can be offset by a lower variance, better MSE, and better out-of-sample performance than the OLS estimates.

The point estimate for ridge regression coefficients is:
$$
\hat{\vec{\beta}}_{\text{ridge}} = {(\mat{X}\T \mat{X} + \lambda \mat{I}_p)}^{-1} \mat{X}\T \vec{y}
$$
The variance-covariance matrix of the point estimate is,
$$
\mathrm{df}(\lambda) = \tr(\mat{X}(\mat{X}\T \mat{X} + \lambda \mat{I}_p)^{-1} \mat{X}\T) = \sum_{j = 1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$
where $d_j$ are the singular values of $X$

Some implications:

- $\hat{\vec{\beta}}$ exists even if $\hat{\vec{\beta}}_{\text{OLS}}$ ($(\mat{X}\T\mat{X})^{-1}$), i.e. cases of $n > p$ and collinearity, does not exist.
- If $\mat{X}$ is orthogonal (mean 0, unit variance, zero correlation), $\mat{X}\T \mat{X} = n \mat{I}_p$ then
  $$
  \hat{\vec{\beta}}_{\text{ridge}} = \frac{n}{n + \lambda} \hat{\vec{\beta}}_{\text{ols}}
  $$
  meaning 
  $$
  |\hat{\vec{\beta}}_{\text{ols}}| > 
  |\hat{\vec{\beta}}_{\text{ridge}}| \geq 0
  $$
- Ridge does not produce sparse estimates, since $(n / (n + \lambda)) \vec{\vec{\beta}}_{ols} = 0$ iff $\vec{\vec{\beta}}_{ols} = 0$
- $\lambda = 0$, then there is no shrinkage
- $\lambda \to \infty$, then there is complete shrinkage and all coefficients are tend to 0.
  
### Lasso

The Lasso or LASSO (least absolute shrinkage and selection operator) replaces squared the penalty on $\beta$ with an absolute value penalty [@Tibshirani1996a]:
$$
\hat{\beta}_{\text{lasso}} = \argmin_{\beta} \frac{1}{2 \sigma} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + \lambda \sum_{k} |\beta_k|
$$
The absolute value penalty will put some $\hat{\beta}_k = 0$, producing a "sparse" solution.


Properties:

- Unlike ridge regression, it sets some coefficients to exactly 0
- If variables are perfectly correlated, there is no unique solution (unlike the ridge regression)
- Used as the best convex approximation of the "best subset selection" regression problem, which finds the number of nonzero entries in a vector.


## Bayesian Shrinkage Priors

$$
\log p(\theta|y, x) \propto \frac{1}{2 \sigma} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + \lambda \sum_{k} \beta_k^2
$$
In the first case, the log density of a normal distribution is,
$$
\log p(y | \mu, x) \propto \frac{1}{2 \sigma} (x - \mu)^2
$$
The first regression term is the produce of normal distributions (sum of their log probabilities),
$$
y_i \sim \dnorm(\vec{x}_i\T \vec{\beta}, \sigma) 
$$
The second term, $\lambda \sum_{k} \beta_k^2$ is also the sum of the log of densities of i.i.d. normal densities, with mean 0, and scale $\tau = 1 / 2 \lambda$,
$$
\beta_k \sim \dnorm(0, \tau^2)
$$

The only difference in the LASSO is the penalty term, which uses an absolute value penalty for $\beta_k$.
That term corresponds to a sum of log densities of i.i.d. double exponential (Laplace) distributions.
The double exponential distribution density is similar to a normal distribution,
$$
\log p(y | \mu, \sigma) \propto - \frac{|y - \mu|}{\sigma}
$$
So the LASSO penalty is equivalent to the log density of a double exponential distribution with location $0$, and scale $1 / \lambda$.
$$
\beta_k \sim \dlaplace(0, \tau)
$$

## Differences between Bayesian Shrinkage and Penalized Likelihood

There are several differences between Bayesian approaches to shrinkage and penalized ML approaches.

The point estimates:

- ML: mode
- Bayesian: posterior mean (or median)

In Lasso

- ML: the mode produces exact zeros and sparsity
- Bayesian: posterior mean is not sparse (zero)

Choosing the shrinkage penalty:

- ML: cross-validation
- Bayesian: a prior is placed on the shrinkage penalty, and it is estimated as part of the posterior.  


## Hierarchical Shrinkage Priors

$$
\begin{aligned}
\beta_k &\sim \dnorm(0, \lambda_i^2 \tau^2) \\
\lambda_i &\sim \dt{\nu}^{+}(0, 1)
\end{aligned}
$$
If $\nu = 1$, then this is the Horseshoe prior
[@CarvalhoPolsonScott2010a, @CarvalhoPolsonScott2009a, @PasKleijnVaart2014a, @DattaGhosh2013a, @PolsonScott2011a, @PiironenVehtari2016a]

Hierarchical Shrinkage Plus (HS-$t_{\nu}$+)

$$
\begin{aligned}
\beta_k &\sim \dnorm(0, \lambda_i^2 \eta_i^2 \tau^2) \\
\lambda_i &\sim \dt{\nu}^{+}(0, 1) \\
\eta_i &\sim \dt{\nu}^{+}(0, 1)
\end{aligned}
$$
This induces even more shrinkage towards zero than the 

If $\nu = 1$, then this is the Horseshoe+ prior as introduced by @BhadraDattaPolsonEtAl2015a.


In linear regression
$$
\begin{aligned}[t]
p(\beta | \Lambda, \tau, \sigma^2, D) &= \dnorm(\beta, \bar{\beta}, \Sigma) \\
\bar{\beta} &= \tau^2 \Lambda (\tau^2 \Lambda + \sigma^2 (X'X)^{-1})^{-1} \hat{\beta} \\
\Sigma &= (\tau^{-2} \Lambda^{-1} + \frac{1}{\sigma^2} X'X)^{-1}
\end{aligned}
$$
where $\Lambda = \diag(\lambda_1^2, \dots, \lambda_D^2)$, and $\hat{\beta}$ is the MLE estimate, $(X'X)^{-1} X' y$.
If predictors are uncorrelated with mean zero and unit variance, then 
$$
X'X \approx n I
$$
and
$$
\bar{\beta}_j = (1 - \kappa_j) \hat{\beta}_j
$$
where
$$
\kappa_j = \frac{1}{1 + n \sigma^{-2} \tau^2 \lambda_j^2}
$$
where $\kappa_j$ is the *shrinkage factor* for the coefficient $\beta_j$, which is how much it is shrunk towards zero from the MLE.
$\kappa_j = 1$ is complete shrinkage, and $\kappa_j = 0$ is no shrinkage.
So $\bar{\beta} \to 0$ as $\tau \to 0$ and $\bar{\beta} \to \hat{\beta}$ as $\tau \to \infty$.

Using a plug-in estimate of $\tau$ using cross-validation or the maximum marginal likelihood. 
The danger is that $\hat{\tau} = 0$ if it is very sparse.

van de Pas et al (2014) show that the optimal value (up to a log factor) in terms of MSE and posterior contraction rates compared to the true $\beta^*$ is
$$
\tau^* = \frac{p^*}{n}
$$
where $p^*$ is the number of non-zero coefficients in the true coefficient vector $\beta^*$.

The effective number of nonzero coefficients is,
$$
m_{\mathrm{eff}} = \sum_{j = 1}^D (1 - \kappa_j)
$$

Some other notes

To calculate the distribution of $\kappa_j$ given a distribution of $\lambda$.
Note that
$$
\kappa_j(\lambda_j) = \frac{1}{1 + n \sigma^{-2} \tau^2 \lambda_j^2}
$$
is monotonically decreasing in $\lambda_j$.
It is also invertible,
$$
\lambda_j(\kappa_j) = \sqrt{\frac{1}{(1 + n \sigma^{-2} \tau^2) \kappa_j}}
$$
The derivative of this with respect to $\kappa_j$ is
$$
\frac{\partial \lambda_j(\kappa_j)}{\partial \kappa_j} = - \sqrt{\frac{1}{(1 + n \sigma^{-2} \tau^2)}} \kappa_j^{-\frac{3}{2}}
$$
The distribution of $\kappa$, given the distribution $f_\lambda$ for lambda is,
$$
\begin{aligned}[t]
f_\kappa(\kappa_j) &= f_\lambda(\lambda_j(\kappa_j)) \left| \frac{\partial \lambda_j(\kappa_j)}{\partial \kappa_j} \right| \\
&= f_\lambda\left(\sqrt{\frac{1}{(1 + n \sigma^{-2} \tau^2) \kappa_j}}\right) \left| (1 + n \sigma^{-2} \tau^2)^{-\frac{1}{2}} \kappa_j^{-\frac{3}{2}} \right| \\
\end{aligned}
$$

Suppose that the distribution is given for precision, $\lambda_j^{-2}$.
Then the inverse is,
$$
\lambda_j^{-2}(\kappa_j) = (1 + n \sigma^{-2} \tau^2) \kappa_j
$$
with derivative,
$$
\frac{\partial \lambda_j^{-2}(\kappa_j)}{\partial \kappa_j} = (1 + n \sigma^{-2} \tau^2)
$$
Thus,
$$
\begin{aligned}[t]
f_\kappa(\kappa_j) &= f_{\lambda^{-2}}(\lambda_j^{-2}(\kappa_j)) \left| \frac{\partial \lambda_j^{-2}(\kappa_j)}{\partial \kappa_j} \right| \\
&= f_{\lambda^{-2}}\left((1 + n \sigma^{-2} \tau^2) \kappa_j \right) \left| (1 + n \sigma^{-2} \tau^2)  \right| \\
\end{aligned}
$$

Suppose that the distribution is given for variance $\lambda_j^2$.
Then the inverse is,
$$
\lambda_j^2(\kappa_j) = \frac{1}{(1 + n \sigma^{-2} \tau^2) \kappa_j}
$$
with derivative,
$$
\frac{\partial \lambda_j^2(\kappa_j)}{\partial \kappa_j} = -(1 + n \sigma^{-2} \tau^2)^{-1} \kappa_j^{-2}
$$
Thus,
$$
\begin{aligned}[t]
f_\kappa(\kappa_j) &= f_{\lambda^2}(\lambda_j^2(\kappa_j)) \left| \frac{\partial \lambda_j^2(\kappa_j)}{\partial \kappa_j} \right| \\
&= f_{\lambda^2}\left(\frac{1}{(1 + n \sigma^{-2} \tau^2) \kappa_j}\right) \left| (1 + n \sigma^{-2} \tau^2)^{-1} \kappa_j^{-2} \right| \\
\end{aligned}
$$



I may also be useful to consider the distribution of $\kappa$ given the distribution of $\tau$.
Note that
$$
\kappa_j(\tau) = \frac{1}{1 + n \sigma^{-2} \tau^2 \lambda_j^2}
$$
is monotonically decreasing in $\tau$.
It is also invertible,
$$
\tau(\kappa_j) = \sqrt{\frac{1}{(1 + n \sigma^{-2} \lambda_j^2) \kappa_j}}
$$
The derivative of this with respect to $\kappa_j$ is
$$
\frac{\partial \tau(\kappa_j)}{\partial \kappa_j} = - {(1 + n \sigma^{-2} \lambda_j^2)}^{-\frac{1}{2}} \kappa_j^{-\frac{3}{2}}
$$
The distribution of $\kappa$, given the distribution $f_\lambda$ for lambda is,
$$
\begin{aligned}[t]
f_\kappa(\kappa_j) &= f_\tau(\tau(\kappa_j)) \left| \frac{\partial \tau(\kappa_j)}{\partial \kappa_j} \right| \\
&= f_\tau\left(\frac{1}{(1 + n \sigma^{-2} \lambda_j^2) \kappa_j} \right) \left| {(1 + n \sigma^{-2} \lambda_j^2)}^{-\frac{1}{2}} \kappa_j^{-\frac{3}{2}} \right| \\
\end{aligned}
$$

- Allan Riddell. [Epistemology of the corral: regression and variable selection with Stan and the Horseshoe prior](https://www.ariddell.org/horseshoe-prior-with-stan.html) March 10, 2014.

## Example

See the [documentation](https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.info.txt).


```r
library("rstan")
library("loo")
library("glmnet")
library("tidyverse")
library("forcats")
library("rubbish")
```



```r
URL <- "https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data"

col_types <- cols(
  X1 = col_integer(),
  lcavol = col_double(),
  lweight = col_double(),
  age = col_integer(),
  lbph = col_double(),
  svi = col_integer(),
  lcp = col_double(),
  gleason = col_integer(),
  pgg45 = col_integer(),
  lpsa = col_double(),
  train = col_logical()
)
prostate <- read_tsv(URL, col_types = col_types,
                     skip = 1,
                     col_names = names(col_types$cols))
```

Recall the prostate data example: we are interested in the level of prostate-specific antigen (PSA), elevated in men who have prostate cancer. 
The data `prostate` has data on on the level of prostate-specific antigen (PSA), which is elevated in men with prostate cancer, for 97 men with
prostate cancer, and clinical predictors. 



```r
f <- lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45 - 1L
```


```r
prostate_data <- lm_preprocess(f, data = prostate)[c("y", "X")] %>%
  within({
    X <- scale(X)
    K <- ncol(X)
    N <- nrow(X)  
  })
```


```r
run_with_tau <- function(tau, mod, data, ...) {
  cat("Tau = ", tau)
  data$tau <- tau
  fit <- sampling(mod, data = data, refresh = -1, verbose = FALSE, ...)
  out <- list()
  out$summary <- summary(fit, par = "b")$summary %>%
    as.data.frame() %>%
    rownames_to_column("parameter")
  
  ## calculate posterior modes
  out$summary$mode <- apply(rstan::extract(fit, "b")[[1]], 2, LaplacesDemon::Mode)
  
  out$summary$tau <- tau
  out$loo <- loo(extract_log_lik(fit))
  out$lppd <- mean(extract_log_lik(fit))
  out$tau <- tau
  out
}
```


```r
mod_lm_coef_normal_1 <- stan_model("stan/lm-coef-normal-1.stan")
```

```r
mod_lm_coef_normal_1
```

<pre>
  <code class="stan">data {
  // number of observations
  int N;
  // response vector
  vector[N] y;
  // number of columns in the design matrix X
  int K;
  // design matrix X
  matrix [N, K] X;
  //
  real<lower = 0.> tau;
}
transformed data {
  real<lower = 0.> y_sd;
  real a_pr_scale;
  real sigma_pr_scale;
  real tau_pr_scale;
  y_sd = sd(y);
  sigma_pr_scale = y_sd * 5.;
  a_pr_scale = 10.;
}
parameters {
  // regression coefficient vector
  real a;
  vector[K] b;
  // scale of the regression errors
  real<lower = 0.> sigma;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[N] mu;
  mu = a + X * b;
}
model {
  // priors
  a ~ normal(0., a_pr_scale);
  b ~ normal(0., tau);
  sigma ~ cauchy(0., sigma_pr_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  // mean log likelihood
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}</code>
</pre>


```r
tau_values <- 2 ^ seq(2, -5, by = -.5)
coefpath_normal <-
  map(tau_values, run_with_tau,
      mod = mod_lm_coef_normal_1, data = prostate_data)
#> Tau =  4
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 0.000104 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 1.04 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.1543 seconds (Warm-up)
#>                0.190139 seconds (Sampling)
#>                0.344439 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.162513 seconds (Warm-up)
#>                0.152582 seconds (Sampling)
#>                0.315095 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.18301 seconds (Warm-up)
#>                0.203906 seconds (Sampling)
#>                0.386916 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.172697 seconds (Warm-up)
#>                0.190956 seconds (Sampling)
#>                0.363653 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  2.83
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.169097 seconds (Warm-up)
#>                0.17704 seconds (Sampling)
#>                0.346137 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.164801 seconds (Warm-up)
#>                0.163248 seconds (Sampling)
#>                0.328049 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.161894 seconds (Warm-up)
#>                0.143494 seconds (Sampling)
#>                0.305388 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.157856 seconds (Warm-up)
#>                0.168858 seconds (Sampling)
#>                0.326714 seconds (Total)
#> 
#> Tau =  2
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.3 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.154067 seconds (Warm-up)
#>                0.160981 seconds (Sampling)
#>                0.315048 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.161518 seconds (Warm-up)
#>                0.171595 seconds (Sampling)
#>                0.333113 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.161746 seconds (Warm-up)
#>                0.174263 seconds (Sampling)
#>                0.336009 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.158683 seconds (Warm-up)
#>                0.183478 seconds (Sampling)
#>                0.342161 seconds (Total)
#> 
#> Tau =  1.41
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.159914 seconds (Warm-up)
#>                0.161926 seconds (Sampling)
#>                0.32184 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.160035 seconds (Warm-up)
#>                0.15434 seconds (Sampling)
#>                0.314375 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 3.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.38 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.176951 seconds (Warm-up)
#>                0.157688 seconds (Sampling)
#>                0.334639 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.171092 seconds (Warm-up)
#>                0.181056 seconds (Sampling)
#>                0.352148 seconds (Total)
#> 
#> Tau =  1
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.27 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.161437 seconds (Warm-up)
#>                0.164732 seconds (Sampling)
#>                0.326169 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.164972 seconds (Warm-up)
#>                0.161937 seconds (Sampling)
#>                0.326909 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.154466 seconds (Warm-up)
#>                0.178186 seconds (Sampling)
#>                0.332652 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.154616 seconds (Warm-up)
#>                0.171853 seconds (Sampling)
#>                0.326469 seconds (Total)
#> 
#> Tau =  0.707
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.31 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.16607 seconds (Warm-up)
#>                0.172031 seconds (Sampling)
#>                0.338101 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.163191 seconds (Warm-up)
#>                0.172667 seconds (Sampling)
#>                0.335858 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.201741 seconds (Warm-up)
#>                0.190361 seconds (Sampling)
#>                0.392102 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.177854 seconds (Warm-up)
#>                0.160295 seconds (Sampling)
#>                0.338149 seconds (Total)
#> 
#> Tau =  0.5
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.158924 seconds (Warm-up)
#>                0.18051 seconds (Sampling)
#>                0.339434 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.175663 seconds (Warm-up)
#>                0.185959 seconds (Sampling)
#>                0.361622 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.4 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.171764 seconds (Warm-up)
#>                0.157551 seconds (Sampling)
#>                0.329315 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.163013 seconds (Warm-up)
#>                0.170233 seconds (Sampling)
#>                0.333246 seconds (Total)
#> 
#> Tau =  0.354
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.153612 seconds (Warm-up)
#>                0.170838 seconds (Sampling)
#>                0.32445 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.150046 seconds (Warm-up)
#>                0.171348 seconds (Sampling)
#>                0.321394 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.155045 seconds (Warm-up)
#>                0.170737 seconds (Sampling)
#>                0.325782 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.27 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.152903 seconds (Warm-up)
#>                0.168739 seconds (Sampling)
#>                0.321642 seconds (Total)
#> 
#> Tau =  0.25
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.4 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.149275 seconds (Warm-up)
#>                0.173242 seconds (Sampling)
#>                0.322517 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.156782 seconds (Warm-up)
#>                0.157241 seconds (Sampling)
#>                0.314023 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.141657 seconds (Warm-up)
#>                0.155932 seconds (Sampling)
#>                0.297589 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.154183 seconds (Warm-up)
#>                0.172394 seconds (Sampling)
#>                0.326577 seconds (Total)
#> 
#> Tau =  0.177
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.143822 seconds (Warm-up)
#>                0.151508 seconds (Sampling)
#>                0.29533 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.179214 seconds (Warm-up)
#>                0.178023 seconds (Sampling)
#>                0.357237 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.138711 seconds (Warm-up)
#>                0.14841 seconds (Sampling)
#>                0.287121 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.14488 seconds (Warm-up)
#>                0.135397 seconds (Sampling)
#>                0.280277 seconds (Total)
#> 
#> Tau =  0.125
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 7.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.71 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.149672 seconds (Warm-up)
#>                0.149073 seconds (Sampling)
#>                0.298745 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.131941 seconds (Warm-up)
#>                0.139512 seconds (Sampling)
#>                0.271453 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.129412 seconds (Warm-up)
#>                0.151603 seconds (Sampling)
#>                0.281015 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.133446 seconds (Warm-up)
#>                0.132621 seconds (Sampling)
#>                0.266067 seconds (Total)
#> 
#> Tau =  0.0884
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.45 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.179764 seconds (Warm-up)
#>                0.121011 seconds (Sampling)
#>                0.300775 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.124772 seconds (Warm-up)
#>                0.107458 seconds (Sampling)
#>                0.23223 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.125093 seconds (Warm-up)
#>                0.10432 seconds (Sampling)
#>                0.229413 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.126442 seconds (Warm-up)
#>                0.108657 seconds (Sampling)
#>                0.235099 seconds (Total)
#> 
#> Tau =  0.0625
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.3 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.149902 seconds (Warm-up)
#>                0.115689 seconds (Sampling)
#>                0.265591 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.122284 seconds (Warm-up)
#>                0.105552 seconds (Sampling)
#>                0.227836 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.118246 seconds (Warm-up)
#>                0.104767 seconds (Sampling)
#>                0.223013 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.115285 seconds (Warm-up)
#>                0.102742 seconds (Sampling)
#>                0.218027 seconds (Total)
#> 
#> Tau =  0.0442
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.34 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.127408 seconds (Warm-up)
#>                0.109975 seconds (Sampling)
#>                0.237383 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.124119 seconds (Warm-up)
#>                0.103228 seconds (Sampling)
#>                0.227347 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.13138 seconds (Warm-up)
#>                0.1137 seconds (Sampling)
#>                0.24508 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.127846 seconds (Warm-up)
#>                0.106882 seconds (Sampling)
#>                0.234728 seconds (Total)
#> 
#> Tau =  0.0312
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.47 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.12768 seconds (Warm-up)
#>                0.108943 seconds (Sampling)
#>                0.236623 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.12821 seconds (Warm-up)
#>                0.113621 seconds (Sampling)
#>                0.241831 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.28 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.118366 seconds (Warm-up)
#>                0.110171 seconds (Sampling)
#>                0.228537 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.121803 seconds (Warm-up)
#>                0.116616 seconds (Sampling)
#>                0.238419 seconds (Total)
```


```r
plot_coefpaths <- function(coefpaths, stat = "mean") {
  ggplot(map_df(coefpaths, "summary"), aes_string(x = "log2(tau)", y = stat,
                       colour = "fct_reorder2(parameter, tau, mean)", fill = "parameter")) +
    modelr::geom_ref_line(h = 0) +
    geom_line() +
    labs(colour = "Parameter")  
}
```

```r
plot_coefpaths(coefpath_normal)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-10-1.png" width="70%" style="display: block; margin: auto;" />



```r
plot_coefpath_loo <- function(x) {
  map_df(x,
       function(x) {
         tibble(tau = x$tau,
                elpd = x$loo$elpd_loo,
                lppd = x$lppd,
                p = x$loo$p_loo)
       }) %>%
    gather(parameter, value, -tau) %>%
    ggplot(aes(x = tau, y = value)) +
    geom_point() +
    geom_line() +
    facet_wrap(~ parameter, scale = "free_y", ncol = 1)
}
```

```r
plot_coefpath_loo(coefpath_normal)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-11-1.png" width="70%" style="display: block; margin: auto;" />


Which is the "best" $tau$?

```r
get_best_tau <- function(coefpath) {
  map_df(coefpath,
       function(x) {
         tibble(tau = x$tau,
                elpd = x$loo$elpd_loo,
                p = x$loo$p_loo)
       }) %>%
    filter(elpd == max(elpd)) 
}
```


```r
get_best_tau(coefpath_normal)
#> # A tibble: 1 × 3
#>     tau  elpd     p
#>   <dbl> <dbl> <dbl>
#> 1  0.25  -108  8.59
```

The mean estimate of $\tau$ is higher than the best estimate, and there is some uncertainty over it. 

```r
mod_lm_coef_normal_2 <- stan_model("stan/lm-coef-normal-2.stan")
#> In file included from filea4941210d6a.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from filea4941210d6a.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from filea4941210d6a.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from filea4941210d6a.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from filea4941210d6a.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/base.hpp:28:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```

```r
fit_normal <- sampling(mod_lm_coef_normal_2, data = prostate_data, refresh = -1,
                 control = list(adapt_delta = 0.99))
#> 
#> Gradient evaluation took 3.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.36 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.356121 seconds (Warm-up)
#>                0.343316 seconds (Sampling)
#>                0.699437 seconds (Total)
#> 
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.355067 seconds (Warm-up)
#>                0.383246 seconds (Sampling)
#>                0.738313 seconds (Total)
#> 
#> 
#> Gradient evaluation took 2.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.24 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.408819 seconds (Warm-up)
#>                0.319976 seconds (Sampling)
#>                0.728795 seconds (Total)
#> 
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.345718 seconds (Warm-up)
#>                0.288584 seconds (Sampling)
#>                0.634302 seconds (Total)
```


```r
summary(fit_normal, "tau")$summary
#>      mean se_mean    sd 2.5%   25%   50%   75% 97.5% n_eff Rhat
#> tau 0.326 0.00203 0.111 0.18 0.251 0.306 0.377   0.6  2983    1
```


```r
loo(extract_log_lik(fit_normal))
#> Computed from 4000 by 97 log-likelihood matrix
#> 
#>          Estimate   SE
#> elpd_loo   -108.9  7.4
#> p_loo         9.2  1.6
#> looic       217.7 14.7
#> 
#> All Pareto k estimates are good (k < 0.5)
#> See help('pareto-k-diagnostic') for details.
```



```r
mcmc_dens(as.array(fit_normal), "tau")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-17-1.png" width="70%" style="display: block; margin: auto;" />


```r
mcmc_dens(as.array(fit_normal), regex_pars = "^b")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-18-1.png" width="70%" style="display: block; margin: auto;" />

### Double Exponential (Laplace) Prior

A second prior to consider for $\vec\beta$ is the Double Exponential.


```r
mod_lasso_1 <- stan_model("stan/lm-coef-lasso-1.stan")
#> In file included from filea496aa58487.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from filea496aa58487.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from filea496aa58487.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from filea496aa58487.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from filea496aa58487.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/base.hpp:28:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
coefpath_lasso <- map(tau_values,
                      run_with_tau,
                   mod = mod_lasso_1,
                   data = prostate_data)
#> Tau =  4
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 6.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.66 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.17492 seconds (Warm-up)
#>                0.203918 seconds (Sampling)
#>                0.378838 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.4 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.23089 seconds (Warm-up)
#>                0.25815 seconds (Sampling)
#>                0.48904 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.249945 seconds (Warm-up)
#>                0.17073 seconds (Sampling)
#>                0.420675 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.160322 seconds (Warm-up)
#>                0.171437 seconds (Sampling)
#>                0.331759 seconds (Total)
#> 
#> Tau =  2.83
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.233266 seconds (Warm-up)
#>                0.226426 seconds (Sampling)
#>                0.459692 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.159963 seconds (Warm-up)
#>                0.178618 seconds (Sampling)
#>                0.338581 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.187203 seconds (Warm-up)
#>                0.236616 seconds (Sampling)
#>                0.423819 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.22078 seconds (Warm-up)
#>                0.174313 seconds (Sampling)
#>                0.395093 seconds (Total)
#> 
#> Tau =  2
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.3 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.152418 seconds (Warm-up)
#>                0.16321 seconds (Sampling)
#>                0.315628 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 3.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.34 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.159017 seconds (Warm-up)
#>                0.154324 seconds (Sampling)
#>                0.313341 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.159111 seconds (Warm-up)
#>                0.162825 seconds (Sampling)
#>                0.321936 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.160985 seconds (Warm-up)
#>                0.174181 seconds (Sampling)
#>                0.335166 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  1.41
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.156407 seconds (Warm-up)
#>                0.180095 seconds (Sampling)
#>                0.336502 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 5.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.55 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.2215 seconds (Warm-up)
#>                0.235111 seconds (Sampling)
#>                0.456611 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.264109 seconds (Warm-up)
#>                0.244515 seconds (Sampling)
#>                0.508624 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.267923 seconds (Warm-up)
#>                0.268041 seconds (Sampling)
#>                0.535964 seconds (Total)
#> 
#> Tau =  1
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 6.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.66 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.163104 seconds (Warm-up)
#>                0.187946 seconds (Sampling)
#>                0.35105 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.163891 seconds (Warm-up)
#>                0.19467 seconds (Sampling)
#>                0.358561 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.195605 seconds (Warm-up)
#>                0.176253 seconds (Sampling)
#>                0.371858 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.29 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.171045 seconds (Warm-up)
#>                0.182622 seconds (Sampling)
#>                0.353667 seconds (Total)
#> 
#> Tau =  0.707
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.29 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.161189 seconds (Warm-up)
#>                0.184951 seconds (Sampling)
#>                0.34614 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.163111 seconds (Warm-up)
#>                0.171172 seconds (Sampling)
#>                0.334283 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.158858 seconds (Warm-up)
#>                0.208449 seconds (Sampling)
#>                0.367307 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.175315 seconds (Warm-up)
#>                0.186435 seconds (Sampling)
#>                0.36175 seconds (Total)
#> 
#> Tau =  0.5
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.47 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.181583 seconds (Warm-up)
#>                0.191976 seconds (Sampling)
#>                0.373559 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.177783 seconds (Warm-up)
#>                0.207356 seconds (Sampling)
#>                0.385139 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.168496 seconds (Warm-up)
#>                0.182827 seconds (Sampling)
#>                0.351323 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 3.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.31 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.21181 seconds (Warm-up)
#>                0.211252 seconds (Sampling)
#>                0.423062 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  0.354
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.33 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.285272 seconds (Warm-up)
#>                0.276128 seconds (Sampling)
#>                0.5614 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 3.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.31 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.241838 seconds (Warm-up)
#>                0.228847 seconds (Sampling)
#>                0.470685 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.280786 seconds (Warm-up)
#>                0.279861 seconds (Sampling)
#>                0.560647 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.201949 seconds (Warm-up)
#>                0.260328 seconds (Sampling)
#>                0.462277 seconds (Total)
#> 
#> Tau =  0.25
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.3 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.158389 seconds (Warm-up)
#>                0.153786 seconds (Sampling)
#>                0.312175 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.24 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.157284 seconds (Warm-up)
#>                0.151697 seconds (Sampling)
#>                0.308981 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.150705 seconds (Warm-up)
#>                0.161812 seconds (Sampling)
#>                0.312517 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.165404 seconds (Warm-up)
#>                0.206648 seconds (Sampling)
#>                0.372052 seconds (Total)
#> 
#> Tau =  0.177
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.38 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.168852 seconds (Warm-up)
#>                0.241641 seconds (Sampling)
#>                0.410493 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.24 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.204063 seconds (Warm-up)
#>                0.166764 seconds (Sampling)
#>                0.370827 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.150153 seconds (Warm-up)
#>                0.187049 seconds (Sampling)
#>                0.337202 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.156256 seconds (Warm-up)
#>                0.189365 seconds (Sampling)
#>                0.345621 seconds (Total)
#> 
#> Tau =  0.125
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.29 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.183602 seconds (Warm-up)
#>                0.242062 seconds (Sampling)
#>                0.425664 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.160867 seconds (Warm-up)
#>                0.161937 seconds (Sampling)
#>                0.322804 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.155386 seconds (Warm-up)
#>                0.165269 seconds (Sampling)
#>                0.320655 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.150882 seconds (Warm-up)
#>                0.162966 seconds (Sampling)
#>                0.313848 seconds (Total)
#> 
#> Tau =  0.0884
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.28 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.156589 seconds (Warm-up)
#>                0.167874 seconds (Sampling)
#>                0.324463 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.169234 seconds (Warm-up)
#>                0.209105 seconds (Sampling)
#>                0.378339 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.24 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.182933 seconds (Warm-up)
#>                0.170653 seconds (Sampling)
#>                0.353586 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.204164 seconds (Warm-up)
#>                0.173697 seconds (Sampling)
#>                0.377861 seconds (Total)
#> 
#> Tau =  0.0625
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.186449 seconds (Warm-up)
#>                0.181862 seconds (Sampling)
#>                0.368311 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.178637 seconds (Warm-up)
#>                0.17492 seconds (Sampling)
#>                0.353557 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.181184 seconds (Warm-up)
#>                0.164903 seconds (Sampling)
#>                0.346087 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.180838 seconds (Warm-up)
#>                0.217337 seconds (Sampling)
#>                0.398175 seconds (Total)
#> 
#> Tau =  0.0442
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.44 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.25347 seconds (Warm-up)
#>                0.248864 seconds (Sampling)
#>                0.502334 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.20956 seconds (Warm-up)
#>                0.182575 seconds (Sampling)
#>                0.392135 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.27 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.226423 seconds (Warm-up)
#>                0.188328 seconds (Sampling)
#>                0.414751 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.211653 seconds (Warm-up)
#>                0.191678 seconds (Sampling)
#>                0.403331 seconds (Total)
#> 
#> Tau =  0.0312
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 2.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.28 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.303936 seconds (Warm-up)
#>                0.293557 seconds (Sampling)
#>                0.597493 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.31577 seconds (Warm-up)
#>                0.223772 seconds (Sampling)
#>                0.539542 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.274582 seconds (Warm-up)
#>                0.281145 seconds (Sampling)
#>                0.555727 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.30202 seconds (Warm-up)
#>                0.367167 seconds (Sampling)
#>                0.669187 seconds (Total)
```


```r
plot_coefpaths(coefpath_lasso)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-21-1.png" width="70%" style="display: block; margin: auto;" />

```r
plot_coefpaths(coefpath_lasso, "mode")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-22-1.png" width="70%" style="display: block; margin: auto;" />


```r
plot_coefpath_pars <- function(coefpath) {
  ggplot(map_df(coefpath, "summary"), aes(x = log10(tau), y = mean)) +
    facet_wrap(~ parameter) +
    modelr::geom_ref_line(h = 0) +
    geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.2) +
    geom_line()  
}
plot_coefpath_pars(coefpath_lasso)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-23-1.png" width="70%" style="display: block; margin: auto;" />


Which is the "best" $tau$?

```r
get_best_tau(coefpath_lasso)
#> # A tibble: 1 × 3
#>     tau  elpd     p
#>   <dbl> <dbl> <dbl>
#> 1  0.25  -108  8.69
```


```r
mod_lasso_2 <- stan_model("stan/lm-coef-lasso-2.stan")
#> In file included from filea495f823861.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from filea495f823861.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from filea495f823861.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from filea495f823861.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from filea495f823861.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/base.hpp:28:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
fit_lasso <- sampling(mod_lasso_2, 
                      data = prostate_data, 
                      refresh = -1,
                      control = list(adapt_delta = 0.9))
#> 
#> Gradient evaluation took 4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.4 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.176768 seconds (Warm-up)
#>                0.174296 seconds (Sampling)
#>                0.351064 seconds (Total)
#> 
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.179308 seconds (Warm-up)
#>                0.164941 seconds (Sampling)
#>                0.344249 seconds (Total)
#> 
#> 
#> Gradient evaluation took 1.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.176375 seconds (Warm-up)
#>                0.166745 seconds (Sampling)
#>                0.34312 seconds (Total)
#> 
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 0.179419 seconds (Warm-up)
#>                0.167266 seconds (Sampling)
#>                0.346685 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 4
#>                                                                                                   count
#> Exception thrown at line 38: double_exponential_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
```


```r
summary(fit_lasso, "tau")$summary
#>      mean se_mean    sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
#> tau 0.263 0.00212 0.122 0.115 0.181 0.236 0.311 0.563  3302    1
```


```r
loo(extract_log_lik(fit_lasso))
#> Computed from 4000 by 97 log-likelihood matrix
#> 
#>          Estimate   SE
#> elpd_loo   -108.3  7.3
#> p_loo         8.7  1.5
#> looic       216.6 14.5
#> 
#> All Pareto k estimates are good (k < 0.5)
#> See help('pareto-k-diagnostic') for details.
```


```r
mcmc_dens(as.array(fit_lasso), "tau")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-29-1.png" width="70%" style="display: block; margin: auto;" />


```r
mcmc_dens(as.array(fit_lasso), regex_pars = "^b")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-30-1.png" width="70%" style="display: block; margin: auto;" />



### Hierarchical Prior (HS)

The Hierarchical or Horseshoe Prior is defined as as a scale mixture of normal distributions,
$$
\begin{aligned}[t]
\lambda_i &\sim \dt{\nu}(0, 1) \\
\end{aligned}
$$
In the original formulation [@CarvalhoPolsonScott2009a,@CarvalhoPolsonScott2010a] use a half-Cauchy ($\nu = 1$), but Stan suggests and **[rstanarm](https://cran.r-project.org/package=rstanarm)** uses 
a Student-t with $\nu = 3$, finding that it has better sampling performance than the half-Cauchy.


```r
mod_lm_coef_hs_1 <- stan_model("stan/lm-coef-hs-1.stan")
#> In file included from filea4957c7f030.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from filea4957c7f030.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from filea4957c7f030.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from filea4957c7f030.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from filea4957c7f030.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/base.hpp:28:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
coefpath_hs <- map(tau_values,
                   run_with_tau, 
                   mod = mod_lm_coef_hs_1,
                   data = c(prostate_data, list(df_local = 3)),
                   control = list(adapt_delta = 0.999, max_treedepth = 12))
#> Tau =  4
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.42 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 8.91271 seconds (Warm-up)
#>                14.1299 seconds (Sampling)
#>                23.0426 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 7.08894 seconds (Warm-up)
#>                5.28367 seconds (Sampling)
#>                12.3726 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 3.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.31 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 8.18893 seconds (Warm-up)
#>                8.57508 seconds (Sampling)
#>                16.764 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.24 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 8.16461 seconds (Warm-up)
#>                9.40632 seconds (Sampling)
#>                17.5709 seconds (Total)
#> Warning: There were 1 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  2.83
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.35 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 6.30514 seconds (Warm-up)
#>                5.20978 seconds (Sampling)
#>                11.5149 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 8.84614 seconds (Warm-up)
#>                7.34652 seconds (Sampling)
#>                16.1927 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 6.14174 seconds (Warm-up)
#>                5.25611 seconds (Sampling)
#>                11.3979 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 6.95079 seconds (Warm-up)
#>                5.98807 seconds (Sampling)
#>                12.9389 seconds (Total)
#> Warning: There were 2 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Tau =  2
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 5.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.51 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 4.90614 seconds (Warm-up)
#>                10.3736 seconds (Sampling)
#>                15.2798 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.6 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 5.95164 seconds (Warm-up)
#>                11.3686 seconds (Sampling)
#>                17.3202 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 4.46233 seconds (Warm-up)
#>                3.38537 seconds (Sampling)
#>                7.8477 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 5.95473 seconds (Warm-up)
#>                4.23735 seconds (Sampling)
#>                10.1921 seconds (Total)
#> Warning: There were 2 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup

#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  1.41
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.43 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 3.85546 seconds (Warm-up)
#>                4.23018 seconds (Sampling)
#>                8.08564 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 4.94961 seconds (Warm-up)
#>                6.84936 seconds (Sampling)
#>                11.799 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 4.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.42 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 4.69521 seconds (Warm-up)
#>                3.66351 seconds (Sampling)
#>                8.35872 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.27 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 4.25596 seconds (Warm-up)
#>                3.88909 seconds (Sampling)
#>                8.14505 seconds (Total)
#> Warning: There were 1 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  1
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 5.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.52 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 3.97371 seconds (Warm-up)
#>                3.63311 seconds (Sampling)
#>                7.60682 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 3.51316 seconds (Warm-up)
#>                3.54133 seconds (Sampling)
#>                7.0545 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.29 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 3.57886 seconds (Warm-up)
#>                3.84504 seconds (Sampling)
#>                7.4239 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 3.57172 seconds (Warm-up)
#>                3.46354 seconds (Sampling)
#>                7.03526 seconds (Total)
#> 
#> Tau =  0.707
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.31 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 3.19481 seconds (Warm-up)
#>                1.68287 seconds (Sampling)
#>                4.87769 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.3 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.43456 seconds (Warm-up)
#>                2.23063 seconds (Sampling)
#>                4.6652 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.70733 seconds (Warm-up)
#>                2.78359 seconds (Sampling)
#>                5.49092 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.62972 seconds (Warm-up)
#>                3.55194 seconds (Sampling)
#>                6.18166 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  0.5
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.31 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.80724 seconds (Warm-up)
#>                2.53693 seconds (Sampling)
#>                5.34417 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.57132 seconds (Warm-up)
#>                4.9641 seconds (Sampling)
#>                7.53542 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 4.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.42 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.52779 seconds (Warm-up)
#>                1.63948 seconds (Sampling)
#>                4.16727 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.47088 seconds (Warm-up)
#>                3.17054 seconds (Sampling)
#>                5.64142 seconds (Total)
#> 
#> Tau =  0.354
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.38 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.37468 seconds (Warm-up)
#>                2.30717 seconds (Sampling)
#>                4.68185 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.72291 seconds (Warm-up)
#>                1.46516 seconds (Sampling)
#>                3.18807 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.08659 seconds (Warm-up)
#>                1.40555 seconds (Sampling)
#>                3.49214 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.9246 seconds (Warm-up)
#>                2.67549 seconds (Sampling)
#>                4.60009 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  0.25
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 6.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.62 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.51227 seconds (Warm-up)
#>                3.22459 seconds (Sampling)
#>                5.73686 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.69852 seconds (Warm-up)
#>                2.68782 seconds (Sampling)
#>                4.38634 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.97475 seconds (Warm-up)
#>                2.75891 seconds (Sampling)
#>                4.73366 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.88592 seconds (Warm-up)
#>                3.05429 seconds (Sampling)
#>                4.9402 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  0.177
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.36 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.86321 seconds (Warm-up)
#>                1.89621 seconds (Sampling)
#>                3.75942 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.18973 seconds (Warm-up)
#>                1.06727 seconds (Sampling)
#>                3.25701 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.53029 seconds (Warm-up)
#>                1.16353 seconds (Sampling)
#>                3.69383 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 3.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.36 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.96126 seconds (Warm-up)
#>                1.57868 seconds (Sampling)
#>                3.53994 seconds (Total)
#> 
#> Tau =  0.125
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.5 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.95433 seconds (Warm-up)
#>                3.05009 seconds (Sampling)
#>                5.00443 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.84622 seconds (Warm-up)
#>                2.08597 seconds (Sampling)
#>                3.93219 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.78112 seconds (Warm-up)
#>                1.51935 seconds (Sampling)
#>                3.30047 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.25613 seconds (Warm-up)
#>                1.19105 seconds (Sampling)
#>                2.44718 seconds (Total)
#> 
#> Tau =  0.0884
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.34 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.24976 seconds (Warm-up)
#>                2.01587 seconds (Sampling)
#>                3.26563 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.44549 seconds (Warm-up)
#>                1.39831 seconds (Sampling)
#>                2.8438 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 3.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.40628 seconds (Warm-up)
#>                0.815435 seconds (Sampling)
#>                2.22171 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.25 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.56705 seconds (Warm-up)
#>                2.20953 seconds (Sampling)
#>                3.77657 seconds (Total)
#> 
#> Tau =  0.0625
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.75582 seconds (Warm-up)
#>                1.85506 seconds (Sampling)
#>                3.61088 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.54255 seconds (Warm-up)
#>                0.837307 seconds (Sampling)
#>                2.37986 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 2.29091 seconds (Warm-up)
#>                2.31218 seconds (Sampling)
#>                4.60309 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.24 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.37782 seconds (Warm-up)
#>                1.33236 seconds (Sampling)
#>                2.71018 seconds (Total)
#> 
#> Tau =  0.0442
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.41546 seconds (Warm-up)
#>                1.49732 seconds (Sampling)
#>                2.91278 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.09049 seconds (Warm-up)
#>                1.82298 seconds (Sampling)
#>                2.91347 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 9.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.91 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.42522 seconds (Warm-up)
#>                1.54792 seconds (Sampling)
#>                2.97314 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.2 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.2673 seconds (Warm-up)
#>                1.35575 seconds (Sampling)
#>                2.62305 seconds (Total)
#> 
#> Tau =  0.0312
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.44 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.56484 seconds (Warm-up)
#>                1.88445 seconds (Sampling)
#>                3.44929 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.36318 seconds (Warm-up)
#>                0.874956 seconds (Sampling)
#>                2.23814 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.14549 seconds (Warm-up)
#>                0.796954 seconds (Sampling)
#>                1.94244 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 2.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 1.62643 seconds (Warm-up)
#>                2.31511 seconds (Sampling)
#>                3.94153 seconds (Total)
#> Warning: There were 1 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
```


```r
plot_coefpaths(coefpath_hs)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-33-1.png" width="70%" style="display: block; margin: auto;" />

```r
plot_coefpaths(coefpath_hs, "mode")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-34-1.png" width="70%" style="display: block; margin: auto;" />

```r
get_best_tau(coefpath_hs)
#> # A tibble: 1 × 3
#>     tau  elpd     p
#>   <dbl> <dbl> <dbl>
#> 1 0.354  -108  8.55
```


```r
plot_coefpath_loo(coefpath_hs)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-36-1.png" width="70%" style="display: block; margin: auto;" />



```r
mod_lm_coef_hs_2 <- stan_model("stan/lm-coef-hs-2.stan")
#> In file included from filea4971f61e44.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from filea4971f61e44.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from filea4971f61e44.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from filea4971f61e44.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from filea4971f61e44.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/base.hpp:28:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
fit_hs <- sampling(mod_lm_coef_hs_2, refresh = -1,
                   data = c(prostate_data, list(df_local = 3, df_global = 3)),
                 control = list(adapt_delta = 0.995))
#> 
#> Gradient evaluation took 4.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.44 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 2.2541 seconds (Warm-up)
#>                1.73173 seconds (Sampling)
#>                3.98583 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 1
#>                                                                          count
#> Exception thrown at line 39: multiply: B[1] is nan, but must not be nan!     2
#> Exception thrown at line 39: multiply: B[8] is nan, but must not be nan!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 2.51608 seconds (Warm-up)
#>                1.80293 seconds (Sampling)
#>                4.31902 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 2
#>                                                                          count
#> Exception thrown at line 39: multiply: B[1] is nan, but must not be nan!     5
#> Exception thrown at line 39: multiply: B[5] is nan, but must not be nan!     1
#> Exception thrown at line 39: multiply: B[7] is nan, but must not be nan!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 1.86659 seconds (Warm-up)
#>                1.49247 seconds (Sampling)
#>                3.35906 seconds (Total)
#> 
#> 
#> Gradient evaluation took 3.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.37 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 1.8229 seconds (Warm-up)
#>                1.5441 seconds (Sampling)
#>                3.367 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 4
#>                                                                          count
#> Exception thrown at line 39: multiply: B[1] is nan, but must not be nan!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> Warning: There were 2 divergent transitions after warmup. Increasing adapt_delta above 0.995 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
```


```r
summary(fit_hs, "tau")$summary
#>      mean se_mean    sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
#> tau 0.365 0.00536 0.213 0.118 0.218 0.315 0.448 0.923  1575    1
```


```r
loo(extract_log_lik(fit_hs))
#> Computed from 4000 by 97 log-likelihood matrix
#> 
#>          Estimate   SE
#> elpd_loo   -108.7  7.1
#> p_loo         8.6  1.4
#> looic       217.3 14.2
#> 
#> All Pareto k estimates are good (k < 0.5)
#> See help('pareto-k-diagnostic') for details.
```


```r
mcmc_dens(as.array(fit_hs), "tau")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-41-1.png" width="70%" style="display: block; margin: auto;" />


```r
mcmc_dens(as.array(fit_hs), regex_pars = "^b\\[\\d+\\]$")
```

<img src="shrinkage_files/figure-html/unnamed-chunk-42-1.png" width="70%" style="display: block; margin: auto;" />


```r
mod_lm_coef_hs_3 <- stan_model("stan/lm-coef-hs-3.stan")
#> In file included from filea4926200396.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from filea4926200396.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from filea4926200396.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from filea4926200396.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from filea4926200396.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/base.hpp:28:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
fit_hs3 <- sampling(mod_lm_coef_hs_3,
                    refresh = -1,
                    data = c(prostate_data, list(df_local = 3, df_global = 3, p0 = 2)),
                    control = list(adapt_delta = 0.995))
#> 
#> Gradient evaluation took 4.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.42 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 6.54601 seconds (Warm-up)
#>                12.1753 seconds (Sampling)
#>                18.7213 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 1
#>                                                                                          count
#> Exception thrown at line 51: student_t_lpdf: Scale parameter is inf, but must be finite!     4
#> Exception thrown at line 44: multiply: B[6] is nan, but must not be nan!                     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> Gradient evaluation took 2.3e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 11.139 seconds (Warm-up)
#>                11.4723 seconds (Sampling)
#>                22.6113 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 2
#>                                                                                          count
#> Exception thrown at line 51: student_t_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> Gradient evaluation took 2.2e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.22 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 7.71319 seconds (Warm-up)
#>                12.6682 seconds (Sampling)
#>                20.3814 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 3
#>                                                                                          count
#> Exception thrown at line 44: multiply: B[1] is nan, but must not be nan!                     1
#> Exception thrown at line 51: student_t_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> Gradient evaluation took 2.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> 
#>  Elapsed Time: 10.1338 seconds (Warm-up)
#>                11.9498 seconds (Sampling)
#>                22.0836 seconds (Total)
#> Warning: There were 133 divergent transitions after warmup. Increasing adapt_delta above 0.995 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: There were 3794 transitions after warmup that exceeded the maximum treedepth. Increase max_treedepth above 10. See
#> http://mc-stan.org/misc/warnings.html#maximum-treedepth-exceeded
#> Warning: Examine the pairs() plot to diagnose sampling problems
```

### Comparison

Let's compare the various coefficient paths:


```r
all_coefpaths <-
  bind_rows(mutate(map_df(coefpath_normal, "summary"), model = "normal"),
          mutate(map_df(coefpath_lasso, "summary"), model = "lasso"),
          mutate(map_df(coefpath_hs, "summary"), model = "hs"))
ggplot(all_coefpaths, aes(x = log2(tau), y = mean, colour = model)) + 
  modelr::geom_ref_line(h = 0) +
  geom_line() +
  facet_wrap(~ parameter)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-45-1.png" width="70%" style="display: block; margin: auto;" />



## Shrinkage Parameters

Given the linear Gaussian regression model
$$
y_i \sim \dnorm(\vec{\beta}\T \vec{x}, \sigma^2)
$$
for $i = 1, \dots, n$, where $\vec{x}$ is the $K$ dimensional vector of predictors.
Suppose a prior
$$
\begin{aligned}[t]
\beta_j | \lambda_j, \tau &\sim \dnorm(0, \lambda_j^2 \tau^2)
\end{aligned}
$$
The $\lambda_j$ are local scales - it allows some weights to escape the shrinkage.
The global parameter $\tau$ pulls all weights towards zero, and effectively controls the sparsity.

The posterior distribution is
$$
\begin{aligned}[t]
p(\vec{\beta} | \mat{\Lambda}, \tau, \sigma^2, \mat{X}, \vec{y}) &= \dnorm(\vec{\beta}, \bar{\vec{\beta}}, \mat{\Sigma}) \\
\bar{\vec{\beta}} &= \tau^2 \mat{\Lambda}(\tau^2 \mat{\Lambda} + \sigma^2 (\mat{X}\T \mat{X})^{-1})^{-1} \hat{\vec{\beta}} \\
\mat{\Sigma} &= (\tau^{-2} \mat{\Lambda}^{-1} + \frac{1}{\sigma^2} \mat{X}\T \mat{X})^{-1},
\end{aligned}
$$
where 
$$
\mat{\Lambda} = \diag(\lambda_1^2, \dots, \lambda_K^2) 
$$
and 
$$
\hat{\vec{\beta}} = (\mat{X}\T \mat{X})^{-1} \mat{X}\T \vec{y}
$$
is the MLE solution if $(\mat{X}\T \mat{X})^{-1}$ exists.

It the predictors are uncorrelated with zero mean and unit variance, then $\mat{X}\T \mat{X} \approx n \mat{I}$, and approximate
$$
\bar{\beta}_j = (1 - \kappa_j) \hat{\beta}_j
$$
where $\kappa_j$ is the shrinkage factor for coefficient $j$,
$$
\kappa_j = \frac{1}{1 + n \sigma^{-2} \tau^2 \lambda^2_j}
$$
When $\kappa = 1$, it is complete shrinkage, and the coefficient is zero.
When $\kappa = 0$, then there is no shrinkage, and the coefficient is equal to the MLE solution.
As $\tau \to 0$, then $\bar{\beta} \to 0$, and as $\tau \to \infty$, then $\bar{\beta} \to \hat{\beta}$.

<img src="shrinkage_files/figure-html/unnamed-chunk-46-1.png" width="70%" style="display: block; margin: auto;" />

Note that for these distributions:

- Normal: prior puts weight only on a single point
- HS for df = 0: prior on shrinkage parameter puts weight on either completely shrunk ($\kappa = 1$) or unshrunk ($\kappa = 0$)
- HS for df = 3: prior on shrinkage parameter puts a lo of weight on it being completely shrunk ($\kappa = 1$), but truncates the density for completely unshrunk.


## Choice of Hyperparameter on $\tau$

The value of $\tau$ and the choice of its hyper-parameter has a big influence on the sparsity of the coefficients.

@CarvalhoPolsonScott2009a suggest 
$$
\tau \sim \dhalfcauchy(0, \sigma),
$$
while @PolsonScott2011a suggest,
$$
\tau \sim \dhalfcauchy(0, 1) .
$$

@PasKleijnVaart2014a suggest 
$$
\tau \sim \dhalfcauchy(0, p^* / n)
$$
where $p^*$ is the true number of non-zero parameters,
and $n$ is the number of observations.
They suggest $\tau = p^* / n$ or $\tau p^*  / n \sqrt{log(n / p^*)}$.
Additionally, they suggest restricting $\tau$ to $[0, 1]$.

@PiironenVehtari2016a understand the choice of the prior on $\tau$ as the implied prior on the number of effective parameters.
The shrinkage can be understood as its influence on the number of effective parameters, $m_{eff}$,
$$
m_{eff} = \sum_{j = 1}^K (1 - \kappa_j) .
$$
This is a measure of effective model size.

The mean and variance of $m_{eff}$ given $\tau$ and $\sigma$ are,
$$
\begin{aligned}[t]
\E[m_{eff} | \tau, \sigma] &= \frac{\sigma^{-1} \tau \sqrt{n}}{1 + \sigma^{-1} \tau \sqrt{n}} K , \\
\Var[m_{eff} | \tau, \sigma] &= \frac{\sigma^{-1} \tau \sqrt{n}}{2 (1 + \sigma^{-1} \tau \sqrt{n})2} K .
\end{aligned}
$$

Based on this, a prior should be chosen so that the prior mass is located near,
$$
\tau_0 = \frac{p_0}{K - p_0}\frac{\sigma}{\sqrt{n}}
$$

Densities of the shrinkage parameter, $\kappa$, for various shrinkage distributions where $\sigma^2 = 1$, $\tau = 1$, for $n = 1$.

@DattaGhosh2013a warn against empirical Bayes estimators of $\tau$ for the horseshoe prior as it can collapse to 0.
@ScottBerger2010a consider marginal maximum likelihood estimates of $\tau$.
@PasKleijnVaart2014a suggest that an empirical Bayes estimator truncated below at $1 / n$.

## R Implementations

See 

- **[rstanarm](https://cran.r-project.org/package=rstanarm)**: estimates GLM regressions with various priors
- **[rmonomvn](https://cran.r-project.org/package=rmonomvn)**: estimates Bayesian ridge, lasso, horseshoe, and ridge regression.
- **[bayesreg](https://cran.r-project.org/package=bayesreg)**: See @MakalicSchmidt2016a for documentation and a good review of Bayesian regularized regression.
- [fastHorseshoe]( http://jingyuhe.com/fastHorseshoe.html)

## Bayesian Model Averaging

*Bayesian model averaging (BMA)* is method that calculates a posterior distribution of parameters by averaging over a discrete set of models, weighting them by their model evidence.

Suppose there are $K$ models, $M_k$, $k = 1, \dots, K$ with the likelihood function $L(y | \theta_k, M_k)$ for observed data $y$.
The posterior distribution of parameters $\theta$, conditional on each model is,
$$
p(\theta_k | y, M_k) = \frac{L(y | \theta_k | M_k) p(\theta_k | M_k)}{\int L(y | \theta_k, M_k) p(\theta_k | M_k) d\,\theta_k}
$$

The essential quality for BMA applications is the denominator of this equation is the the *marginal likelihood* or *model evidence*,
$$
p(y | M_k) = \int L(y | \theta_k, M_k) p(\theta_k | M_k) d\,\theta_k .
$$

From this, derive the posterior probability of models given the data,
$$
p(M_k | y) = \frac{p(y | M_k) p(M_k)}{\sum_{m = 1}^K p(y | M_m) p(M_m)}
$$
The posterior probability of a model requires specifying a prior $p(M_k)$ for each model.

Bayes Factors can be used to calculate model probabilities for BMA and vice-versa.
The Bayes Factor for models $l$ and $m$ is
$$
BF_{lm} = \frac{p(M_l | y)}{p(M_m | y)} .
$$
Given a baseline model, $M_1$, the model evidence can be written in terms of Bayes Factors relative to that model,
$$
p(M_l|y) = \frac{BF_{1l} p(M_l)}{\sum_{m = 1}^K BF_{m1} p(M_m)} .
$$

**marginal probabilities of a parameter:** The marginal probability of a parameter ($\theta$), averaged across all models is,
$$
p(\theta | y) = \sum_{k = 1}^K p(\theta | y, M_k) p(M_k | y) .
$$
The posterior distribution of $\Delta$ averaged across all models is the average of $\Delta$ weighted by each posterior model probability.
The mean and variance of the posterior models are,
$$
\begin{aligned}[t]
\E[\theta | y] &= \sum_{k = 1}^K \bar{\theta} p(M_k | y) \\
\Var[\theta | y] &= \sum_{k = 1}^K 
(\Var(\theta | y, M_k) + \bar{\theta}_k^2) p(M_k | y) - \E(\theta | y)^2
\end{aligned}
$$


Why is BMA difficult? 

- The posterior is sensitive to the model prior, $p(M_k)$.
- Calculating the model evidence, $p(y | M_k)$, is computationally difficult, except in special cases
- The model space can be very large. In regression, it is $2^K$. This means that it may be impossible to compute model probabilities for the full set of models. Thus, it may require sampling from the (discrete) model space.

Uses of BMA:

1. model selection or choice: select the most likely model
2. average posterior estimates
3. average prediction. Generally predictions from models using BMA have lower risk (Raftery) 

For the common case of linear regression,
$$
\begin{aligned}[t]
y &= \alpha + X \beta + \epsilon & \epsilon &\sim \dnorm(0, \sigma^2 I)
\end{aligned}
$$
where $X$ is a $N \times K$ matrix and $\beta$ is a $K \times 1$ vector.
The model selection problem in this case is the choice of the $K$ variables to include in the regression model.
Thus, there are $2^K$ models to consider.
Very quickly, 

See @FragosoNeto2015a for a recent review. See @VolinskyRafteryMadiganEtAl1999a for an earlier review.

There are several R packages that implement BMA. See @AminiShahramParmeterChristopher2011a for a review of R packages.

- **[BAS](https://cran.r-project.org/package=BAS)** See its vignette @Zeugner2011a.
- **[BMA](https://cran.r-project.org/package=BMA)** See its vignette @RafteryHoetingVolinskyEtAl2017a.
- **[BMS](https://cran.r-project.org/package=BMS)** See its vignette @Clyde2017a.
- **[ensembleBMA](https://cran.r-project.org/package=ensembleBMA)** uses *BMA* to generates ensemble BMA forecasts

### Zellner's g-prior

An alternative prior is the Zellner's g-prior.
Consider the regression,
$$
y_i | \alpha, \vec{\beta}, \sigma \sim \dnorm(\alpha + \mat{X} \vec{\beta}, \sigma^2)
$$
The $g$-prior is a non-informative, data-dependent prior,
$$
\vec{\beta} \sim \dnorm(0, \sigma^2 g \mat{X}\T \mat{X})
$$
It depends on only a single parameter $g$.
The prior for $g$ must be proper. Some common choices include,
$$
\begin{aligned}
g &= n \\
g &= k^2 \\
g &= \max(n, k^2)
\end{aligned}
$$
or putting a hyperprior on $g$.

See @LeySteel2012a for a recent overview of g-priors.


## Slab and Spike Priors

In the case of the linear regression, an alternative to BMA is to use a spike-and-slab prior [@MitchellBeauchamp1988a, @GeorgeMcCulloch1993a, @IshwaranRao2005a], which is a prior that is a discrete mixture of a point mass at 0 and a non-informative distribution. The weight over these who alternatives is similar to a 

The spike and slab prior is a "two-group" solution

$$
p(\beta_k) = (1 - w) \delta_0 + w \pi(\beta_k)
$$
where $\delta_0$ is a Dirac delta function putting a point mass at 0, and $\pi(\beta_k)$ is an uninformative distribution, e.g. $\pi(\beta_k) = \dnorm(\beta_k | 0, \sigma^2)$ where $\sigma$ is large.

The posterior distribution of $w$ is the probability that $\beta_k \neq 0$, and the conditional posterior distribution $p(\beta_k | y, w = 1)$ is the distribution of $\beta_k$ given that $\beta_k \neq 0$.

See the R package **[spikeslab](https://cran.r-project.org/package=spikeslab)** and he accompanying article [@IshwaranKogalurRao2010a] for an implementation and review of spike-and-slab regressions.

## Technical Notes

Marginal density of the horseshoe+ prior @CarvalhoPolsonScott2010a has no closed form but some bounds
are available.
If $\tau^2 = 1$, then the marginal density of the horseshoe+ prior has the following properties:
$$
\begin{aligned}[t]
\frac{K}{2} \log \left(1 + \frac{4}{\theta^2} \right) < p_{HS}(\theta) \leq K \log \left(1 + \frac{2}{\theta^2} \right) \\
\lim_{|\theta| \to 0} p_{HS}(\theta) = \infty
\end{aligned}
$$
where $K = 1 / \sqrt{2 \pi^3}$.

Marginal density of the horseshoe+ prior @BhadraDattaPolsonEtAl2015a:
If $\tau^2 = 1$, then the marginal density of the horseshoe+ prior has the following properties:
$$
\begin{aligned}[t]
\frac{1}{\pi^2 \sqrt{2 \pi}} \log \left(1 + \frac{4}{\theta^2} \right) < p_{HS+}(\theta) \leq \frac{1}{\pi^2 |\theta|} \\
\lim_{|\theta| \to 0} p_{HS+}(\theta) = \infty
\end{aligned}
$$


Prior for $\theta_i$               Density for $\lambda_i$                      Density for $\kappa_i$
---------------------------------- -------------------------------------------- -------------------------------------------------------------------------------
Double-exponential                 $\lambda_i \exp(\lambda_i^2 / 2)             $\kappa_i^{-2} \exp\left( \frac{- 1}{2 \kappa_i} \right)$
Cauchy                             $\lambda_i^{-2} \exp(-1 / \lambda_i^2)$      $\kappa_i^{-\frac{1}{2}} (1 - \kappa_i)^{- \frac{3}{2}} \exp \left(\frac{\kappa_i}{2 (1 - \kappa_i)}\right)$
Strawderman-Berger                 $\lambda_i (1 + \lambda_i^2)^{-\frac{3}{2}}$ $\kappa_i^{-\frac{1}{2}}$
Normal-exponential-gamma           $\lambda_i (1 + \lambda_i^2)^{-(c + 1)}$     $\kappa_i^{c - 1}$
Normal-Jeffreys                    $1 / \lambda_i$                              $\kappa_i^{-1} (1 - \kappa_i)^{-1}$
Horseshoe                           $(1 + \lambda_i^2)^{-1}$                     $\kappa_i^{-1/2} (1 - \kappa_i)^{-1/2}$


Thresh-holding. The horseshoe has an implicit threshold of $|T_\tau(y) - y| < \sqrt{2 \sigma ^ 2 \log (1 / \tau))$ [@PasKleijnVaart2014a].


## Multiple Comparisons and Thresholding rules

Multiple comparisons, family-wise error rate, and false discovery rates are frequentist 
concepts. There are some attempts to bridge these two worlds - see Efron in particular.  However, even if methodologically different, shrinkage addresses some of broadest
concerns about making multiple comparisons.

Although discussing hierarchical models, @GelmanHillYajima2012a compares the shrinkage in hierarchical models to multiple comparisons, also see this [post](http://andrewgelman.com/2013/08/20/correcting-for-multiple-comparisons-in-a-bayesian-regression-model/).

Another (related) issue is sparsification. The decision rule as to whether a variable
is 0 (included), or not.

- The sparse-shrinkage priors from @CarvalhoPolsonScott2010a are motivated by a two-group
model (either $\beta = 0$ or $\beta \neq 0$). They suggest a decision rule of considering
$\beta \neq 0$ when $E(\kappa_j) < 0.5$ where $\kappa_j$ is a shrinkage parameter described in the paper.
- @HahnCarvalho2015a propose estimating the posterior distribution via shrinkage, and then summarizing the posterior distribution.
- @PiironenVehtari2015a propose something similar in spirit, in which a second step projects the initial shrinkage model to a sparse model

## Examples of Applications of Sensitivity Analysis

These are a few applied papers based on sensitivity or model analysis meaning primarily problem, rather than methodologically driven.

The memorably titled "Let's Take the Con Out of Econometrics" [@Leamer1983a],
that economic models fail to account for model uncertainty and suggests using 
an ensemble-like method called extreme-bounds.

On economic growth: (confidence bounds) @Sala-I-Martin1997a, (Bayesian model averaging) @FernandezLeySteel2001a, @LeySteel2009a, @EicherPapageorgiouRaftery2009a, @BrockDurlaufWest2003a

Wars: @HegreSambanis2006a use extreme bounds for civil war onset. @WardGreenhillBakke2010a use model comparison and a step-wise method, but are focused on the difference between note the difference between p-values and prediction; @Goenner2004a use BMA for inter-state wars (democratic peace). 

@MontgomeryHollenbachWard2012a and @MontgomeryNyhan2010a apply BMA to multiple political science issues including voting, presidential elections, and civil war onset.

@TobiasLi2004a Returns to Schooling

See @FragosoNeto2015a for a recent(ish) review of BMA applications.

Also, not that many of these analyses are slightly older as empirical research 
in economics and political science has put less emphasis on model-based inference 
(all-cause regressions) and more on design-based (causal) inference methods
As noted earlier, regularization techniques are also applicable in these cases, but 
need to be adjusted.
