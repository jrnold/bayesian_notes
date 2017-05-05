
Consider the single output linear Gaussian regression model with several input variables, given by
$$
\begin{aligned}[t]
y_i \sim N(\vec{x}_i' \vec{\beta}, \sigma^2)
\end{aligned}
$$
where $\vec{x}$ is a $k$-vector of predictors, and $\vec{\beta}$ are the coefficients.

What priors do we put on $\beta$? 

- **Improproper priors:** $\beta_k \propto 1$ This produces the equivalent of MLE estimates.
- **Non-informative priors:** These are priors which have such wide variance that they have little influence on the posterior, e.g. $\beta_k \sim N(0, 1e6)$. The primary reason for these (as opposed to simply using an improper prior) is that some MCMC methods, e.g. Gibbs sampling as used in JAGS or BUGS, require proper prior distributions for all parameters.
- **Weakly informative priors:** This is described in @GelmanJakulinPittauEtAl2008a, @Betancourt2017a, and are the defaults suggested in **[rstanarm](https://cran.r-project.org/package=rstanarm)**. Examples would include $\beta_k \sim \dnorm(0, 2.5)$ or $\beta_k \sim \dcauchy(0, 2.5)$. The purpose of these priors is to ex ante rule out implausible priors. This can help with computation, produce proper posterior distributions, and can help with some estimation issues such as separation or quasi-separation in discrete response models. These priors may help reduce the variance of posterior distribution estimates, they generally are intended to be weak enough to not affect the posterior mean point estimates too much.

**Shrinkage priors** have a couple characteristics

- they push $\beta_k \to 0$
- while in the other cases, the scale of the prior on $\beta$ is fixed, in shrinkage priors there is often a hyperprior on it. E.g. $\beta_k \sim N(0, \tau)$, where $\tau$ is also a parameter to be estimated.

Penalized Regression
$$
\hat{\beta}_{penalized} = \argmin_{\beta} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + f(\beta)
$$
where $f$ is some sort of penalty function on $\beta$ that penalizes larger (in magnitude) values of $\beta$.

Two common penalized regressions are lasso and ridge.

Ridge Regression

$$
\hat{\beta}_{ridge} = \argmin_{\beta} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + \lambda \sum_{k} \beta_k^2
$$
This penalty produces smaller in magnitude coefficients, $|\hat{\beta}_{ridge}| < |\hat{\beta}_{OLS}|$.
However, this "bias" in the coefficients can be offset by a lower variance, better MSE, and better out-of-sample performance than the OLS estimates.

LASSO: The LASSO replaces squared the penalty on $\beta$ with an absolute value penalty.
$$
\hat{\beta}_{lasso} = \argmin_{\beta} \frac{1}{2 \sigma} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + \lambda \sum_{k} |\beta_k|
$$
The absolute value penalty will put some $\hat{\beta}_k = 0$, producing a "sparse" solution.


## Bayesian formulation

$$
\log p(\theta|y, x) \propto \frac{1}{2 \sigma} \sum_{i = 1}^n (\vec{x}_i\T \vec{\beta} - y_i)^2 + \lambda \sum_{k} \beta_k^2
$$
In the first case, the log density of a normal distribution is,
$$
\log p(y | \mu, x) \propto \frac{1}{2 \sigma} (x - \mu)^2
$$
The first regression term is the produce of normal distributions (sum of their log probabilities),
$$
y_i \sim N(\vec{x}_i\T \vec{\beta}, \sigma) 
$$
The second term, $\lambda \sum_{k} \beta_k^2$ is also the sum of the log of densities of i.i.d. normal densities, with mean 0, and scale $\tau = 1 / 2 \lambda$,
$$
\beta_k \sim N(0, \tau^2)
$$

The only difference in the LASSO is the penalty term, which uses an absolute value penalty for $\beta_k$.
That term corresponds to a sum of log densities of i.i.d. double exponential (Laplace) distributions.
The double exponential distribution density is similar to a normal distribution,
$$
\log p(y | \mu, \sigma) \propto - \frac{|y - \mu|}{\sigma}
$$
So the LASSO penalty is equaivalent to the log density of a double exponential distribution with location $0$, and scale $1 / \lambda$.
$$
\beta_k \sim \dlaplace(0, \tau)
$$


There are several differences between Bayesian approaches to shrinkage and penalized ML approaches.

The most important is the primary point estimate.
In penalized ML, the point estimate is the mode. 
In full Bayesian estimation, the primary point estimate of interest is the mean. 
This difference will be most apparent in sparse shrinkage esimators such as the LASSO.

Using a double exponential penalty induces a sparse solution for $\hat\beta$ by placing the modes of the coefficients at zero. 
However, using the double exponential as a prior does not place the means of coefficients at zero.

Comparing the normal and double exponential distributions: 

1. the double exponential distribution has a spike at zero, unlike the normal
2. however, the probability mass of these distributions (important for the posterior mean) is not too different.


Difference

- The point estimates are the posterior modes. In Bayesian estimation,
    posterior means are the preferred point estimate.
- The shrinkage parameter $\lambda$ is often a chosen by cross-validation
    or an approximation thereof. In Bayesian estimation, the estimation 
    marginalizes over uncertainty about the shrinkage parameter.



Hierarchical Shrinkage (HS-$t_{\nu}$) 

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

@PiironenVehtari2015b, "Projection predictive variable selection using Stan+R"


In linear regression
$$
\begin{aligned}[t]
p(\beta | \Lambda, \tau, \sigma^2, D) &= N(\beta, \bar{\beta}, \Sigma) \\
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

Using a plug-in estimate of $\tau$ using cross-validatio or max marginal likelihood. 
The danger is that $\hat{\tau} = 0$ if it is very sparse.

van de Pas et al (2014) show that the optimal value (upp to a log factor) in terms of MSE and posterior contraction rates compared to the true $\beta^*$ is
$$
\tau^* = \frac{p^*}{n}
$$
where $p^*$ is the number of non-zero coefficients in the true coefficient vector $\beta^*$.

The effective number of nonzero coefficients is,
$$
m_{eff} = \sum_{j = 1}^D (1 - \kappa_j)
$$


- Allan Riddell. [Epistemology of the corral: regression and variable selection with Stan and the Horseshoe prior](https://www.ariddell.org/horseshoe-prior-with-stan.html) March 10, 2014.

## Worked Example

See the [documentation](https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.info.txt).


```r
library("rstan")
library("loo")
library("glmnet")
library("tidyverse")
library("forcats")
library("rubbish")
```


### Example


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
  fit <- sampling(mod, data = data, open_progress = FALSE, 
                   show_messages = FALSE, verbose = FALSE, ...)
  out <- list()
  out$summary <- summary(fit, par = "b")$summary %>%
    as.data.frame() %>%
    rownames_to_column("parameter")
  
  ## calculate posterior modes
  out$summary$mode <- apply(rstan::extract(fit, "b")[[1]], 2, LaplacesDemon::Mode)
  
  out$summary$tau <- tau
  out$loo <- loo(extract_log_lik(fit))
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
  mu = X * b;
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
#>  Elapsed Time: 0.192392 seconds (Warm-up)
#>                0.132851 seconds (Sampling)
#>                0.325243 seconds (Total)
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
#>  Elapsed Time: 0.154875 seconds (Warm-up)
#>                0.139629 seconds (Sampling)
#>                0.294504 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.184782 seconds (Warm-up)
#>                0.148844 seconds (Sampling)
#>                0.333626 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.182544 seconds (Warm-up)
#>                0.149316 seconds (Sampling)
#>                0.33186 seconds (Total)
#> 
#> Tau =  2.83
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
#>  Elapsed Time: 0.180457 seconds (Warm-up)
#>                0.15334 seconds (Sampling)
#>                0.333797 seconds (Total)
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
#>  Elapsed Time: 0.171127 seconds (Warm-up)
#>                0.194767 seconds (Sampling)
#>                0.365894 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.173013 seconds (Warm-up)
#>                0.140324 seconds (Sampling)
#>                0.313337 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.181726 seconds (Warm-up)
#>                0.138685 seconds (Sampling)
#>                0.320411 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
#> Tau =  2
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.167481 seconds (Warm-up)
#>                0.134794 seconds (Sampling)
#>                0.302275 seconds (Total)
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
#>  Elapsed Time: 0.171777 seconds (Warm-up)
#>                0.132113 seconds (Sampling)
#>                0.30389 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.19115 seconds (Warm-up)
#>                0.150624 seconds (Sampling)
#>                0.341774 seconds (Total)
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
#>  Elapsed Time: 0.199805 seconds (Warm-up)
#>                0.149285 seconds (Sampling)
#>                0.34909 seconds (Total)
#> 
#> Tau =  1.41
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
#>  Elapsed Time: 0.157746 seconds (Warm-up)
#>                0.131732 seconds (Sampling)
#>                0.289478 seconds (Total)
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
#>  Elapsed Time: 0.169804 seconds (Warm-up)
#>                0.153345 seconds (Sampling)
#>                0.323149 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.182874 seconds (Warm-up)
#>                0.129109 seconds (Sampling)
#>                0.311983 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.177958 seconds (Warm-up)
#>                0.1425 seconds (Sampling)
#>                0.320458 seconds (Total)
#> 
#> Tau =  1
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.160393 seconds (Warm-up)
#>                0.140111 seconds (Sampling)
#>                0.300504 seconds (Total)
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
#>  Elapsed Time: 0.153314 seconds (Warm-up)
#>                0.131865 seconds (Sampling)
#>                0.285179 seconds (Total)
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
#>  Elapsed Time: 0.160682 seconds (Warm-up)
#>                0.119537 seconds (Sampling)
#>                0.280219 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.178966 seconds (Warm-up)
#>                0.143555 seconds (Sampling)
#>                0.322521 seconds (Total)
#> 
#> Tau =  0.707
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.153218 seconds (Warm-up)
#>                0.13677 seconds (Sampling)
#>                0.289988 seconds (Total)
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
#>  Elapsed Time: 0.150144 seconds (Warm-up)
#>                0.130474 seconds (Sampling)
#>                0.280618 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.169275 seconds (Warm-up)
#>                0.145652 seconds (Sampling)
#>                0.314927 seconds (Total)
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
#>  Elapsed Time: 0.169707 seconds (Warm-up)
#>                0.13395 seconds (Sampling)
#>                0.303657 seconds (Total)
#> 
#> Tau =  0.5
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.133876 seconds (Warm-up)
#>                0.111592 seconds (Sampling)
#>                0.245468 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.150307 seconds (Warm-up)
#>                0.13838 seconds (Sampling)
#>                0.288687 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.162937 seconds (Warm-up)
#>                0.118181 seconds (Sampling)
#>                0.281118 seconds (Total)
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
#>  Elapsed Time: 0.147555 seconds (Warm-up)
#>                0.122727 seconds (Sampling)
#>                0.270282 seconds (Total)
#> 
#> Tau =  0.354
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
#>  Elapsed Time: 0.142904 seconds (Warm-up)
#>                0.098279 seconds (Sampling)
#>                0.241183 seconds (Total)
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
#>  Elapsed Time: 0.174574 seconds (Warm-up)
#>                0.117813 seconds (Sampling)
#>                0.292387 seconds (Total)
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
#>  Elapsed Time: 0.131847 seconds (Warm-up)
#>                0.091777 seconds (Sampling)
#>                0.223624 seconds (Total)
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
#>  Elapsed Time: 0.142174 seconds (Warm-up)
#>                0.096783 seconds (Sampling)
#>                0.238957 seconds (Total)
#> 
#> Tau =  0.25
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.131232 seconds (Warm-up)
#>                0.085902 seconds (Sampling)
#>                0.217134 seconds (Total)
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
#>  Elapsed Time: 0.115831 seconds (Warm-up)
#>                0.09366 seconds (Sampling)
#>                0.209491 seconds (Total)
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
#>  Elapsed Time: 0.132924 seconds (Warm-up)
#>                0.095038 seconds (Sampling)
#>                0.227962 seconds (Total)
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
#>  Elapsed Time: 0.133025 seconds (Warm-up)
#>                0.088969 seconds (Sampling)
#>                0.221994 seconds (Total)
#> 
#> Tau =  0.177
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
#>  Elapsed Time: 0.174173 seconds (Warm-up)
#>                0.095372 seconds (Sampling)
#>                0.269545 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.116576 seconds (Warm-up)
#>                0.084611 seconds (Sampling)
#>                0.201187 seconds (Total)
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
#>  Elapsed Time: 0.136271 seconds (Warm-up)
#>                0.087522 seconds (Sampling)
#>                0.223793 seconds (Total)
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
#>  Elapsed Time: 0.173263 seconds (Warm-up)
#>                0.098489 seconds (Sampling)
#>                0.271752 seconds (Total)
#> 
#> Tau =  0.125
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.41 seconds.
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
#>  Elapsed Time: 0.137786 seconds (Warm-up)
#>                0.087603 seconds (Sampling)
#>                0.225389 seconds (Total)
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
#>  Elapsed Time: 0.134625 seconds (Warm-up)
#>                0.088345 seconds (Sampling)
#>                0.22297 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.128999 seconds (Warm-up)
#>                0.085438 seconds (Sampling)
#>                0.214437 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.132792 seconds (Warm-up)
#>                0.087709 seconds (Sampling)
#>                0.220501 seconds (Total)
#> 
#> Tau =  0.0884
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
#>  Elapsed Time: 0.150915 seconds (Warm-up)
#>                0.09087 seconds (Sampling)
#>                0.241785 seconds (Total)
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
#>  Elapsed Time: 0.131397 seconds (Warm-up)
#>                0.098308 seconds (Sampling)
#>                0.229705 seconds (Total)
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
#>  Elapsed Time: 0.142437 seconds (Warm-up)
#>                0.085196 seconds (Sampling)
#>                0.227633 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.136415 seconds (Warm-up)
#>                0.090986 seconds (Sampling)
#>                0.227401 seconds (Total)
#> 
#> Tau =  0.0625
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.39 seconds.
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
#>  Elapsed Time: 0.162939 seconds (Warm-up)
#>                0.093922 seconds (Sampling)
#>                0.256861 seconds (Total)
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
#>  Elapsed Time: 0.160651 seconds (Warm-up)
#>                0.089396 seconds (Sampling)
#>                0.250047 seconds (Total)
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
#>  Elapsed Time: 0.176672 seconds (Warm-up)
#>                0.102306 seconds (Sampling)
#>                0.278978 seconds (Total)
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
#>  Elapsed Time: 0.153515 seconds (Warm-up)
#>                0.089334 seconds (Sampling)
#>                0.242849 seconds (Total)
#> 
#> Tau =  0.0442
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.144152 seconds (Warm-up)
#>                0.088395 seconds (Sampling)
#>                0.232547 seconds (Total)
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
#>  Elapsed Time: 0.153132 seconds (Warm-up)
#>                0.09676 seconds (Sampling)
#>                0.249892 seconds (Total)
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
#>  Elapsed Time: 0.164349 seconds (Warm-up)
#>                0.088316 seconds (Sampling)
#>                0.252665 seconds (Total)
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
#>  Elapsed Time: 0.1541 seconds (Warm-up)
#>                0.260077 seconds (Sampling)
#>                0.414177 seconds (Total)
#> 
#> Tau =  0.0312
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 3.7e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.37 seconds.
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
#>  Elapsed Time: 0.190448 seconds (Warm-up)
#>                0.091445 seconds (Sampling)
#>                0.281893 seconds (Total)
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
#>  Elapsed Time: 0.223739 seconds (Warm-up)
#>                0.099944 seconds (Sampling)
#>                0.323683 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.211555 seconds (Warm-up)
#>                0.092466 seconds (Sampling)
#>                0.304021 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.207275 seconds (Warm-up)
#>                0.084787 seconds (Sampling)
#>                0.292062 seconds (Total)
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
#> 1 0.177  -234   2.3
```

The mean estimate of $\tau$ is higher than the best estimate, and there is some uncertainty over it. 

```r
mod_lm_coef_normal_2 <- stan_model("stan/lm-coef-normal-2.stan")
#> In file included from file858e6c44a278.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config.hpp:39:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file858e6c44a278.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file858e6c44a278.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file858e6c44a278.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from file858e6c44a278.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/base.hpp:28:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```

```r
fit_normal <- sampling(mod_lm_coef_normal_2, data = prostate_data,
                 control = list(adapt_delta = 0.95))
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-2' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.225052 seconds (Warm-up)
#>                0.895729 seconds (Sampling)
#>                1.12078 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-2' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.4e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.14 seconds.
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
#>  Elapsed Time: 0.259847 seconds (Warm-up)
#>                0.166059 seconds (Sampling)
#>                0.425906 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-2' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.220589 seconds (Warm-up)
#>                0.156875 seconds (Sampling)
#>                0.377464 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-normal-2' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.281895 seconds (Warm-up)
#>                0.216293 seconds (Sampling)
#>                0.498188 seconds (Total)
#> Warning: There were 1 divergent transitions after warmup. Increasing adapt_delta above 0.95 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
```


```r
summary(fit_normal, "tau")$summary
#>      mean se_mean    sd   2.5%   25%   50%  75% 97.5% n_eff Rhat
#> tau 0.257 0.00639 0.142 0.0418 0.158 0.238 0.33 0.599   498    1
```


```r
loo(extract_log_lik(fit_normal))
#> Computed from 4000 by 97 log-likelihood matrix
#> 
#>          Estimate  SE
#> elpd_loo   -234.7 3.0
#> p_loo         3.7 0.4
#> looic       469.3 6.1
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
#> In file included from file858e1884d9ba.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config.hpp:39:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file858e1884d9ba.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file858e1884d9ba.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file858e1884d9ba.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from file858e1884d9ba.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/base.hpp:28:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
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
#>  Elapsed Time: 0.203125 seconds (Warm-up)
#>                0.219432 seconds (Sampling)
#>                0.422557 seconds (Total)
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
#>  Elapsed Time: 0.226381 seconds (Warm-up)
#>                0.147806 seconds (Sampling)
#>                0.374187 seconds (Total)
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
#>  Elapsed Time: 0.174772 seconds (Warm-up)
#>                0.136878 seconds (Sampling)
#>                0.31165 seconds (Total)
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
#>  Elapsed Time: 0.232236 seconds (Warm-up)
#>                0.219128 seconds (Sampling)
#>                0.451364 seconds (Total)
#> 
#> Tau =  2.83
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.295551 seconds (Warm-up)
#>                0.154536 seconds (Sampling)
#>                0.450087 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.162578 seconds (Warm-up)
#>                0.130441 seconds (Sampling)
#>                0.293019 seconds (Total)
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
#>  Elapsed Time: 0.179926 seconds (Warm-up)
#>                0.20317 seconds (Sampling)
#>                0.383096 seconds (Total)
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
#>  Elapsed Time: 0.181609 seconds (Warm-up)
#>                0.138067 seconds (Sampling)
#>                0.319676 seconds (Total)
#> 
#> Tau =  2
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.244725 seconds (Warm-up)
#>                0.158295 seconds (Sampling)
#>                0.40302 seconds (Total)
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
#>  Elapsed Time: 0.171891 seconds (Warm-up)
#>                0.142987 seconds (Sampling)
#>                0.314878 seconds (Total)
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
#>  Elapsed Time: 0.180301 seconds (Warm-up)
#>                0.136516 seconds (Sampling)
#>                0.316817 seconds (Total)
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
#>  Elapsed Time: 0.176716 seconds (Warm-up)
#>                0.122646 seconds (Sampling)
#>                0.299362 seconds (Total)
#> 
#> Tau =  1.41
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.161651 seconds (Warm-up)
#>                0.145732 seconds (Sampling)
#>                0.307383 seconds (Total)
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
#>  Elapsed Time: 0.181198 seconds (Warm-up)
#>                0.126522 seconds (Sampling)
#>                0.30772 seconds (Total)
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
#>  Elapsed Time: 0.161207 seconds (Warm-up)
#>                0.139827 seconds (Sampling)
#>                0.301034 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.204387 seconds (Warm-up)
#>                0.185243 seconds (Sampling)
#>                0.38963 seconds (Total)
#> 
#> Tau =  1
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.168931 seconds (Warm-up)
#>                0.124869 seconds (Sampling)
#>                0.2938 seconds (Total)
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
#>  Elapsed Time: 0.167754 seconds (Warm-up)
#>                0.166777 seconds (Sampling)
#>                0.334531 seconds (Total)
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
#>  Elapsed Time: 0.178087 seconds (Warm-up)
#>                0.129965 seconds (Sampling)
#>                0.308052 seconds (Total)
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
#>  Elapsed Time: 0.151388 seconds (Warm-up)
#>                0.135604 seconds (Sampling)
#>                0.286992 seconds (Total)
#> 
#> Tau =  0.707
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.48 seconds.
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
#>  Elapsed Time: 0.169259 seconds (Warm-up)
#>                0.17781 seconds (Sampling)
#>                0.347069 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.194361 seconds (Warm-up)
#>                0.146883 seconds (Sampling)
#>                0.341244 seconds (Total)
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
#>  Elapsed Time: 0.171286 seconds (Warm-up)
#>                0.147479 seconds (Sampling)
#>                0.318765 seconds (Total)
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
#>  Elapsed Time: 0.153118 seconds (Warm-up)
#>                0.125549 seconds (Sampling)
#>                0.278667 seconds (Total)
#> 
#> Tau =  0.5
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.161618 seconds (Warm-up)
#>                0.131861 seconds (Sampling)
#>                0.293479 seconds (Total)
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
#>  Elapsed Time: 0.179635 seconds (Warm-up)
#>                0.139476 seconds (Sampling)
#>                0.319111 seconds (Total)
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
#>  Elapsed Time: 0.178076 seconds (Warm-up)
#>                0.141196 seconds (Sampling)
#>                0.319272 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.159317 seconds (Warm-up)
#>                0.128316 seconds (Sampling)
#>                0.287633 seconds (Total)
#> 
#> Tau =  0.354
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
#>  Elapsed Time: 0.175414 seconds (Warm-up)
#>                0.144926 seconds (Sampling)
#>                0.32034 seconds (Total)
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
#>  Elapsed Time: 0.182694 seconds (Warm-up)
#>                0.103902 seconds (Sampling)
#>                0.286596 seconds (Total)
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
#>  Elapsed Time: 0.163483 seconds (Warm-up)
#>                0.118778 seconds (Sampling)
#>                0.282261 seconds (Total)
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
#>  Elapsed Time: 0.164961 seconds (Warm-up)
#>                0.142643 seconds (Sampling)
#>                0.307604 seconds (Total)
#> 
#> Tau =  0.25
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.187546 seconds (Warm-up)
#>                0.144197 seconds (Sampling)
#>                0.331743 seconds (Total)
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
#>  Elapsed Time: 0.199597 seconds (Warm-up)
#>                0.162147 seconds (Sampling)
#>                0.361744 seconds (Total)
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
#>  Elapsed Time: 0.181932 seconds (Warm-up)
#>                0.230224 seconds (Sampling)
#>                0.412156 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.350175 seconds (Warm-up)
#>                0.16838 seconds (Sampling)
#>                0.518555 seconds (Total)
#> 
#> Tau =  0.177
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.233327 seconds (Warm-up)
#>                0.16501 seconds (Sampling)
#>                0.398337 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.19735 seconds (Warm-up)
#>                0.149318 seconds (Sampling)
#>                0.346668 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.235574 seconds (Warm-up)
#>                0.250456 seconds (Sampling)
#>                0.48603 seconds (Total)
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
#>  Elapsed Time: 0.231094 seconds (Warm-up)
#>                0.16289 seconds (Sampling)
#>                0.393984 seconds (Total)
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
#>  Elapsed Time: 0.3106 seconds (Warm-up)
#>                0.247362 seconds (Sampling)
#>                0.557962 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.395404 seconds (Warm-up)
#>                0.225327 seconds (Sampling)
#>                0.620731 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.397722 seconds (Warm-up)
#>                0.199453 seconds (Sampling)
#>                0.597175 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.32251 seconds (Warm-up)
#>                0.193525 seconds (Sampling)
#>                0.516035 seconds (Total)
#> 
#> Tau =  0.0884
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
#>  Elapsed Time: 0.324556 seconds (Warm-up)
#>                0.211534 seconds (Sampling)
#>                0.53609 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.315736 seconds (Warm-up)
#>                0.218104 seconds (Sampling)
#>                0.53384 seconds (Total)
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
#>  Elapsed Time: 0.35155 seconds (Warm-up)
#>                0.22798 seconds (Sampling)
#>                0.57953 seconds (Total)
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
#>  Elapsed Time: 0.355479 seconds (Warm-up)
#>                0.22844 seconds (Sampling)
#>                0.583919 seconds (Total)
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
#>  Elapsed Time: 0.336196 seconds (Warm-up)
#>                0.248033 seconds (Sampling)
#>                0.584229 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.280733 seconds (Warm-up)
#>                0.180144 seconds (Sampling)
#>                0.460877 seconds (Total)
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
#>  Elapsed Time: 0.36307 seconds (Warm-up)
#>                0.208637 seconds (Sampling)
#>                0.571707 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.301935 seconds (Warm-up)
#>                0.271584 seconds (Sampling)
#>                0.573519 seconds (Total)
#> 
#> Tau =  0.0442
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 6.8e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.68 seconds.
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
#>  Elapsed Time: 0.308225 seconds (Warm-up)
#>                0.196725 seconds (Sampling)
#>                0.50495 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 1.5e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
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
#>  Elapsed Time: 0.307672 seconds (Warm-up)
#>                0.184948 seconds (Sampling)
#>                0.49262 seconds (Total)
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
#>  Elapsed Time: 0.337673 seconds (Warm-up)
#>                0.344071 seconds (Sampling)
#>                0.681744 seconds (Total)
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
#>  Elapsed Time: 0.388897 seconds (Warm-up)
#>                0.192425 seconds (Sampling)
#>                0.581322 seconds (Total)
#> 
#> Tau =  0.0312
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
#>  Elapsed Time: 0.422058 seconds (Warm-up)
#>                0.279718 seconds (Sampling)
#>                0.701776 seconds (Total)
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
#>  Elapsed Time: 0.406669 seconds (Warm-up)
#>                0.261016 seconds (Sampling)
#>                0.667685 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.486101 seconds (Warm-up)
#>                0.317999 seconds (Sampling)
#>                0.8041 seconds (Total)
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
#>  Elapsed Time: 0.330692 seconds (Warm-up)
#>                0.315719 seconds (Sampling)
#>                0.646411 seconds (Total)
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
#> 1 0.125  -234  2.47
```


```r
mod_lasso_2 <- stan_model("stan/lm-coef-lasso-2.stan")
#> In file included from file858e723827e0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config.hpp:39:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file858e723827e0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file858e723827e0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file858e723827e0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from file858e723827e0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/base.hpp:28:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
fit_lasso <- sampling(mod_lasso_2, data = prostate_data,
                 control = list(adapt_delta = 0.9))
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-2' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.715269 seconds (Warm-up)
#>                0.343596 seconds (Sampling)
#>                1.05886 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-2' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.941103 seconds (Warm-up)
#>                0.497253 seconds (Sampling)
#>                1.43836 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-2' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.749805 seconds (Warm-up)
#>                0.310293 seconds (Sampling)
#>                1.0601 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-lasso-2' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.804896 seconds (Warm-up)
#>                0.371644 seconds (Sampling)
#>                1.17654 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 4
#>                                                                                                   count
#> Exception thrown at line 38: double_exponential_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> Warning: There were 1 chains where the estimated Bayesian Fraction of Missing Information was low. See
#> http://mc-stan.org/misc/warnings.html#bfmi-low
#> Warning: Examine the pairs() plot to diagnose sampling problems
```


```r
summary(fit_lasso, "tau")$summary
#>      mean se_mean    sd   2.5%  25%   50%   75% 97.5% n_eff Rhat
#> tau 0.216 0.00859 0.146 0.0173 0.12 0.193 0.282 0.576   288 1.01
```


```r
loo(extract_log_lik(fit_lasso))
#> Computed from 4000 by 97 log-likelihood matrix
#> 
#>          Estimate  SE
#> elpd_loo   -234.6 3.0
#> p_loo         3.7 0.4
#> looic       469.2 6.0
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

The Hierarchical or Horseshoe Prior is defined as as a scale mixture of normals,
$$
\begin{aligned}[t]
\lambda_i &\sim \dt{\nu}(0, 1) \\
\end{aligned}
$$
In the original formulation [@CarvalhoPolsonScott2009a,@CarvalhoPolsonScott2010a] use a half-Cauchy ($\nu = 1$), but Stan suggests and **[rstanarm](https://cran.r-project.org/package=rstanarm)** uses 
a Student-t with $\nu = 3$, finding that it has better sampling performance than the half-Cauchy.





```r
mod_lm_coef_hs_1 <- stan_model("stan/lm-coef-hs-1.stan")
#> In file included from file858e4cd1ce0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config.hpp:39:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file858e4cd1ce0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file858e4cd1ce0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file858e4cd1ce0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from file858e4cd1ce0.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/base.hpp:28:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
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
#>  Elapsed Time: 3.0942 seconds (Warm-up)
#>                6.656 seconds (Sampling)
#>                9.7502 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 3.63222 seconds (Warm-up)
#>                1.66763 seconds (Sampling)
#>                5.29985 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 3.09333 seconds (Warm-up)
#>                2.43475 seconds (Sampling)
#>                5.52807 seconds (Total)
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
#>  Elapsed Time: 3.21133 seconds (Warm-up)
#>                3.84505 seconds (Sampling)
#>                7.05638 seconds (Total)
#> Warning: There were 2 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Tau =  2.83
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
#>  Elapsed Time: 2.80535 seconds (Warm-up)
#>                4.9817 seconds (Sampling)
#>                7.78705 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 2.32679 seconds (Warm-up)
#>                1.87897 seconds (Sampling)
#>                4.20576 seconds (Total)
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
#>  Elapsed Time: 2.50865 seconds (Warm-up)
#>                2.77957 seconds (Sampling)
#>                5.28822 seconds (Total)
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
#>  Elapsed Time: 2.32385 seconds (Warm-up)
#>                3.54379 seconds (Sampling)
#>                5.86764 seconds (Total)
#> 
#> Tau =  2
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 2.64403 seconds (Warm-up)
#>                1.1519 seconds (Sampling)
#>                3.79593 seconds (Total)
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
#>  Elapsed Time: 1.86939 seconds (Warm-up)
#>                1.14436 seconds (Sampling)
#>                3.01374 seconds (Total)
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
#>  Elapsed Time: 1.94494 seconds (Warm-up)
#>                2.25723 seconds (Sampling)
#>                4.20216 seconds (Total)
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
#>  Elapsed Time: 1.90254 seconds (Warm-up)
#>                1.22839 seconds (Sampling)
#>                3.13094 seconds (Total)
#> Warning: There were 2 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup

#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Tau =  1.41
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 1.33148 seconds (Warm-up)
#>                1.29478 seconds (Sampling)
#>                2.62626 seconds (Total)
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
#>  Elapsed Time: 1.91548 seconds (Warm-up)
#>                1.24393 seconds (Sampling)
#>                3.1594 seconds (Total)
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
#>  Elapsed Time: 2.20486 seconds (Warm-up)
#>                0.777258 seconds (Sampling)
#>                2.98212 seconds (Total)
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
#>  Elapsed Time: 2.37431 seconds (Warm-up)
#>                0.704522 seconds (Sampling)
#>                3.07883 seconds (Total)
#> Warning: There were 3 divergent transitions after warmup. Increasing adapt_delta above 0.999 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup

#> Warning: Examine the pairs() plot to diagnose sampling problems
#> Tau =  1
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
#>  Elapsed Time: 1.486 seconds (Warm-up)
#>                1.50744 seconds (Sampling)
#>                2.99345 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 1.41879 seconds (Warm-up)
#>                1.96182 seconds (Sampling)
#>                3.38061 seconds (Total)
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
#>  Elapsed Time: 1.28532 seconds (Warm-up)
#>                0.830735 seconds (Sampling)
#>                2.11606 seconds (Total)
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
#>  Elapsed Time: 1.76351 seconds (Warm-up)
#>                1.27295 seconds (Sampling)
#>                3.03646 seconds (Total)
#> 
#> Tau =  0.707
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
#>  Elapsed Time: 1.21697 seconds (Warm-up)
#>                1.5279 seconds (Sampling)
#>                2.74487 seconds (Total)
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
#>  Elapsed Time: 1.29116 seconds (Warm-up)
#>                1.68829 seconds (Sampling)
#>                2.97946 seconds (Total)
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
#>  Elapsed Time: 0.975277 seconds (Warm-up)
#>                0.856215 seconds (Sampling)
#>                1.83149 seconds (Total)
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
#>  Elapsed Time: 1.54229 seconds (Warm-up)
#>                0.749365 seconds (Sampling)
#>                2.29166 seconds (Total)
#> 
#> Tau =  0.5
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
#>  Elapsed Time: 0.9807 seconds (Warm-up)
#>                0.664109 seconds (Sampling)
#>                1.64481 seconds (Total)
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
#>  Elapsed Time: 0.873667 seconds (Warm-up)
#>                0.704927 seconds (Sampling)
#>                1.57859 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 3).
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
#>  Elapsed Time: 1.59464 seconds (Warm-up)
#>                0.537478 seconds (Sampling)
#>                2.13212 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 4.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.41 seconds.
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
#>  Elapsed Time: 0.969919 seconds (Warm-up)
#>                0.852639 seconds (Sampling)
#>                1.82256 seconds (Total)
#> 
#> Tau =  0.354
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
#>  Elapsed Time: 0.966356 seconds (Warm-up)
#>                1.30417 seconds (Sampling)
#>                2.27053 seconds (Total)
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
#>  Elapsed Time: 1.05749 seconds (Warm-up)
#>                1.23329 seconds (Sampling)
#>                2.29078 seconds (Total)
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
#>  Elapsed Time: 0.888172 seconds (Warm-up)
#>                0.657467 seconds (Sampling)
#>                1.54564 seconds (Total)
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
#>  Elapsed Time: 0.813759 seconds (Warm-up)
#>                0.619169 seconds (Sampling)
#>                1.43293 seconds (Total)
#> 
#> Tau =  0.25
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 1.03296 seconds (Warm-up)
#>                0.671805 seconds (Sampling)
#>                1.70477 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.877272 seconds (Warm-up)
#>                0.682668 seconds (Sampling)
#>                1.55994 seconds (Total)
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
#>  Elapsed Time: 0.755829 seconds (Warm-up)
#>                0.646209 seconds (Sampling)
#>                1.40204 seconds (Total)
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
#>  Elapsed Time: 0.891953 seconds (Warm-up)
#>                0.638756 seconds (Sampling)
#>                1.53071 seconds (Total)
#> 
#> Tau =  0.177
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 1.45331 seconds (Warm-up)
#>                0.398344 seconds (Sampling)
#>                1.85165 seconds (Total)
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
#>  Elapsed Time: 0.764744 seconds (Warm-up)
#>                0.79891 seconds (Sampling)
#>                1.56365 seconds (Total)
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
#>  Elapsed Time: 1.41794 seconds (Warm-up)
#>                1.14188 seconds (Sampling)
#>                2.55982 seconds (Total)
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
#>  Elapsed Time: 0.696781 seconds (Warm-up)
#>                0.769285 seconds (Sampling)
#>                1.46607 seconds (Total)
#> 
#> Tau =  0.125
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
#>  Elapsed Time: 1.28076 seconds (Warm-up)
#>                0.479483 seconds (Sampling)
#>                1.76024 seconds (Total)
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
#>  Elapsed Time: 0.8366 seconds (Warm-up)
#>                0.400772 seconds (Sampling)
#>                1.23737 seconds (Total)
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
#>  Elapsed Time: 0.931097 seconds (Warm-up)
#>                0.647405 seconds (Sampling)
#>                1.5785 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 1.00569 seconds (Warm-up)
#>                0.977764 seconds (Sampling)
#>                1.98346 seconds (Total)
#> 
#> Tau =  0.0884
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
#>  Elapsed Time: 0.956376 seconds (Warm-up)
#>                0.662644 seconds (Sampling)
#>                1.61902 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.639587 seconds (Warm-up)
#>                0.679683 seconds (Sampling)
#>                1.31927 seconds (Total)
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
#>  Elapsed Time: 0.673781 seconds (Warm-up)
#>                0.450492 seconds (Sampling)
#>                1.12427 seconds (Total)
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
#>  Elapsed Time: 0.762565 seconds (Warm-up)
#>                0.673966 seconds (Sampling)
#>                1.43653 seconds (Total)
#> 
#> Tau =  0.0625
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
#>  Elapsed Time: 0.636234 seconds (Warm-up)
#>                1.22508 seconds (Sampling)
#>                1.86132 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.646499 seconds (Warm-up)
#>                0.333274 seconds (Sampling)
#>                0.979773 seconds (Total)
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
#>  Elapsed Time: 0.636346 seconds (Warm-up)
#>                0.660609 seconds (Sampling)
#>                1.29696 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.54939 seconds (Warm-up)
#>                0.464511 seconds (Sampling)
#>                1.0139 seconds (Total)
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
#>  Elapsed Time: 1.06192 seconds (Warm-up)
#>                0.486088 seconds (Sampling)
#>                1.54801 seconds (Total)
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
#>  Elapsed Time: 0.525964 seconds (Warm-up)
#>                0.581703 seconds (Sampling)
#>                1.10767 seconds (Total)
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
#>  Elapsed Time: 0.58582 seconds (Warm-up)
#>                0.627907 seconds (Sampling)
#>                1.21373 seconds (Total)
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
#>  Elapsed Time: 0.584653 seconds (Warm-up)
#>                0.552962 seconds (Sampling)
#>                1.13761 seconds (Total)
#> 
#> Tau =  0.0312
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.9558 seconds (Warm-up)
#>                0.445267 seconds (Sampling)
#>                1.40107 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.740804 seconds (Warm-up)
#>                0.495384 seconds (Sampling)
#>                1.23619 seconds (Total)
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
#>  Elapsed Time: 0.643832 seconds (Warm-up)
#>                0.346732 seconds (Sampling)
#>                0.990564 seconds (Total)
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
#>  Elapsed Time: 0.622372 seconds (Warm-up)
#>                0.672157 seconds (Sampling)
#>                1.29453 seconds (Total)
#> Warning: Some Pareto k diagnostic values are slightly high. See
#> help('pareto-k-diagnostic') for details.
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
#> 1 0.125  -234  2.51
```


```r
plot_coefpath_loo(coefpath_hs)
```

<img src="shrinkage_files/figure-html/unnamed-chunk-36-1.png" width="70%" style="display: block; margin: auto;" />



```r
mod_lm_coef_hs_2 <- stan_model("stan/lm-coef-hs-2.stan")
#> In file included from file858e7f4a443e.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config.hpp:39:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file858e7f4a443e.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file858e7f4a443e.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file858e7f4a443e.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from file858e7f4a443e.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/base.hpp:28:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
fit_hs <- sampling(mod_lm_coef_hs_2, data = c(prostate_data, list(df_local = 3, df_global = 3)),
                 control = list(adapt_delta = 0.995))
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-2' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 4.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.41 seconds.
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
#>  Elapsed Time: 1.05013 seconds (Warm-up)
#>                0.602248 seconds (Sampling)
#>                1.65238 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 1
#>                                                                          count
#> Exception thrown at line 39: multiply: B[1] is nan, but must not be nan!     2
#> Exception thrown at line 39: multiply: B[8] is nan, but must not be nan!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-2' NOW (CHAIN 2).
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
#>  Elapsed Time: 1.02277 seconds (Warm-up)
#>                0.444002 seconds (Sampling)
#>                1.46678 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 2
#>                                                                          count
#> Exception thrown at line 39: multiply: B[1] is nan, but must not be nan!     5
#> Exception thrown at line 39: multiply: B[5] is nan, but must not be nan!     1
#> Exception thrown at line 39: multiply: B[7] is nan, but must not be nan!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-2' NOW (CHAIN 3).
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
#>  Elapsed Time: 1.20737 seconds (Warm-up)
#>                1.22761 seconds (Sampling)
#>                2.43498 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-2' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.991417 seconds (Warm-up)
#>                0.495344 seconds (Sampling)
#>                1.48676 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 4
#>                                                                          count
#> Exception thrown at line 39: multiply: B[1] is nan, but must not be nan!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
```


```r
summary(fit_hs, "tau")$summary
#>      mean se_mean    sd   2.5%   25%  50%   75% 97.5% n_eff Rhat
#> tau 0.298 0.00459 0.223 0.0361 0.147 0.24 0.382 0.885  2359    1
```


```r
loo(extract_log_lik(fit_hs))
#> Computed from 4000 by 97 log-likelihood matrix
#> 
#>          Estimate  SE
#> elpd_loo   -234.5 3.0
#> p_loo         3.7 0.4
#> looic       469.0 5.9
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
#> In file included from file858e35b5bcd3.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config.hpp:39:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/config/compiler/clang.hpp:196:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file858e35b5bcd3.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:42:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file858e35b5bcd3.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core.hpp:43:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file858e35b5bcd3.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:59:
#> /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> In file included from file858e35b5bcd3.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/rev/mat.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/mat.hpp:298:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr.hpp:39:
#> In file included from /Users/jrnold/Library/R/3.3/library/StanHeaders/include/stan/math/prim/arr/functor/integrate_ode_rk45.hpp:13:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint.hpp:61:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/numeric/odeint/util/multi_array_adaption.hpp:29:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array.hpp:21:
#> In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/base.hpp:28:
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:42:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:43:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:53:43: warning: unused typedef 'index_range' [-Wunused-local-typedef]
#>       typedef typename Array::index_range index_range;
#>                                           ^
#> /Library/Frameworks/R.framework/Versions/3.3/Resources/library/BH/include/boost/multi_array/concept_checks.hpp:54:37: warning: unused typedef 'index' [-Wunused-local-typedef]
#>       typedef typename Array::index index;
#>                                     ^
#> 8 warnings generated.
```


```r
fit_hs3 <- sampling(mod_lm_coef_hs_3, data = c(prostate_data, list(df_local = 3, df_global = 3, p0 = 2)),
                 control = list(adapt_delta = 0.995))
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-3' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 6.9e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.69 seconds.
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
#>  Elapsed Time: 8.63179 seconds (Warm-up)
#>                11.9353 seconds (Sampling)
#>                20.5671 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 1
#>                                                                                          count
#> Exception thrown at line 51: student_t_lpdf: Scale parameter is inf, but must be finite!     4
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-3' NOW (CHAIN 2).
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
#>  Elapsed Time: 7.35979 seconds (Warm-up)
#>                11.7959 seconds (Sampling)
#>                19.1557 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-3' NOW (CHAIN 3).
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
#>  Elapsed Time: 10.234 seconds (Warm-up)
#>                11.9845 seconds (Sampling)
#>                22.2185 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 3
#>                                                                                          count
#> Exception thrown at line 44: multiply: B[1] is nan, but must not be nan!                     1
#> Exception thrown at line 51: student_t_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'lm-coef-hs-3' NOW (CHAIN 4).
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
#>  Elapsed Time: 3.17774 seconds (Warm-up)
#>                6.45774 seconds (Sampling)
#>                9.63548 seconds (Total)
#> Warning: There were 531 divergent transitions after warmup. Increasing adapt_delta above 0.995 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: There were 3342 transitions after warmup that exceeded the maximum treedepth. Increase max_treedepth above 10. See
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
