
# Generalized Linear Models

## Generalized Linear Models 

Generalized linear models (GLMs) are a class of commonly used models.[^glm-r]
In GLMs, the mean is specified as a function of a linear model of predictors,
$$
E(Y) = \mu = g^{-1}(\mat{X} \vec{\beta}) .
$$
GLMs are a generalization of linear regression from an unbounded continuous outcome variable to other types of data: binary, count, categorical, bounded continuous.

A GLM consists of three components:

1. A *probability distribution* (*family*) specifying the conditional distribution of the response variable.
    In GLMs, the distribution is in the exponential family: Normal, Binomial, Poisson, Categorical, Multinomial, Poisson, Beta.
2. A *linear predictor*, which is a linear function of the predictors,
    $$
    \eta = \mat{X} \vec{\beta} .
    $$
3. A *link function* ($g(.)$) which maps the expected value to the  the linear predictor,
    $$
    g(\mu) = \eta .
    $$
    The link function is smooth and invertible, and the  *inverse link function* or *mean function* maps the linear predictor to the mean,
    $$
    \mu = g^{-1}(\eta) .
    $$
    The link function ($g$) and its inverse ($g^{-1}) translate $\eta$ from $(\-infty, +\infty)$ to the proper range for the probability distribution and back again.

These models are often estimated with MLE, as with the function [stats](https://www.rdocumentation.org/packages/stats/topics/glm). 
However, these are also easily estimated in a Bayesian setting.

See the help for [stats](https://www.rdocumentation.org/packages/stats/topics/family) for common probability distributions, [stats](https://www.rdocumentation.org/packages/stats/topics/make.link) for common links,  and the [Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model) page for a table of common GLMs.
See the function **[VGAM](https://cran.r-project.org/package=VGAM)** for even more examples of link functions and probability distributions.

Link                       Range of $\mu_i$                    $\eta_i = g(\mu_i)$                         $\mu_i = g^{-1}(\eta)_i$
-------------------------- ----------------------------------- ------------------------------------------- ----------------------------------------
Identity                   $(-\infty, \infty)$                 $\mu_i$                                     $\eta_i$
Inverse                    $(-\infty, \infty) \setminus \{0\}$ $\mu_i^{-1}$                                $\eta_i^{-1}$
Log                        $(0, \infty)$                       $\log(\mu_i)$                               $\exp(\eta_i)$
Inverse-square             $(0, \infty)$                       $\mu_i^{-2}$                                $\eta_i^{-1/2}$
Square-root                $(0, \infty)$                       $\sqrt{\mu_i}$                              $\eta_{i}^2$
Logit                      $(0, 1)$                            $\log(\mu / (1 - \mu_i)$                    $1 / (1 + \exp(-\eta_i))$
Probit                     $(0, 1)$                            $\Phi^{-1}(\mu_i)$                          $\Phi(\eta_i)$
Cauchit                    $(0, 1)$                            $\tan(\pi (\mu_i - 1 / 2))$                 $\frac{1}{\pi} \arctan(\eta_i) + \frac{1}{2}$
Log-log                    $(0, 1)$                            $-\log(-log(\mu_i))$                        $\exp(-\exp(-\eta_i))$
Complementary Log-log      $(0, 1)$                            $\log(-log(1 - \mu_i))$                     $1 - \exp(-\exp(\eta_i))$

Table:  Common Link Functions and their inverses. Table derived from @Fox2016a [p. 419].


Distribution           Canonical Link  Range of $Y_i$                                                        Other link functions
---------------------- --------------- --------------------------------------------------------------------- ------------------------------
Normal                 Identity        real: $(-\infty, +\infty)$                                            log, inverse
Exponential            Inverse         real: $(0, +\infty)$                                                  identity, log
Gamma                  Inverse         real: $(0, +\infty)$                                                  identity, log
Inverse-Gaussian       Inverse-squared real: $(0, +\infty)$                                                  inverse, identity, log
Bernoulli              Logit           integer: $\{0, 1\}$                                                   probit, cauchit, log, cloglog
Binomial               Logit           integer: $0, 1, \dots, n_i$                                           probit, cauchit, log, cloglog
Poisson                Log             integer: $0, 1, 2, \dots$                                             identity, sqrt
Categorical            Logit           $0, 1, \dots, K$                                                      
Multinomial            Logit           K-vector of integers, $\{x_1, \dots, x_K\}$ s.t. $\sum_k x_k = N$.

Table: Common distributions and link functions. Table derived from @Fox2016a [p. 421],  [Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model), and [stats](https://www.rdocumentation.org/packages/stats/topics/glm).


## Count Models

### Poisson 

The Poisson model is used for unbounded count data,
$$
Y = 0, 1, \dots, \infty
$$
The outcome is modeled as a Poisson distribution
$$
y_i \sim \dpois(\lambda_i)
$$
with positive mean parameter $\lambda_i \in (0, \infty)$.
Since $\lambda_i$ has to be positive, the most common link function is the log,
$$
\log(\lambda_i) = \exp(\vec{x}_i' \vec{\beta})
$$
which has the inverse,
$$
\lambda_i = \log(\vec{x}_i \vec{\beta})
$$

In Stan, the Poisson distribution has two implementations:

- `poisson_lpdf`
- `poisson_log_lpdf`: Poisson with a log link. This is for numeric stability.

Also, `rstanarm` supports the [Poisson](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html).

## Example

A regression model of bilateral sanctions for the period 1939 to 1983. 
The outcome variable is the number of countries imposing sanctions.

```r
data("sanction", package = "Zelig")
```



```r
library("rstan")
library("tidyverse")
library("magrittr")
#> 
#> Attaching package: 'magrittr'
#> The following object is masked from 'package:purrr':
#> 
#>     set_names
#> The following object is masked from 'package:tidyr':
#> 
#>     extract
#> The following object is masked from 'package:rstan':
#> 
#>     extract

URL <- "https://raw.githubusercontent.com/carlislerainey/priors-for-separation/master/br-replication/data/need.csv"

autoscale <- function(x, center = TRUE, scale = TRUE) {
  nvals <- length(unique(x))
  if (nvals <= 1) {
    out <- x
  } else if (nvals == 2) {
    out <- if (scale) {
      (x - min(x, na.rm = TRUE)) / diff(range(x, finite = TRUE))
    } else x
    if (center) {
      out <- x - mean(x)
    }
  } else {
    out <- if (center) {
      x - mean(x, na.rm = TRUE)
    } else x
    out <- if (scale) out / sd(out, na.rm = TRUE)
  }
  out
}


f <- (oppose_expansion ~ dem_governor + obama_win + gop_leg + percent_uninsured +
      income + percent_nonwhite + percent_metro)

br <- read_csv(URL) %>%
  mutate(oppose_expansion = 1 - support_expansion,
         dem_governor = -1 * gop_governor,
         obama_win = as.integer(obama_share >= 0.5),
         percent_nonwhite = percent_black + percent_hispanic) %>%
  rename(gop_leg = legGOP) %>%
  # keep only variables in the formula
  model.frame(f, data = .) %>%
  # drop missing values (if any?)
  drop_na()
#> Parsed with column specification:
#> cols(
#>   .default = col_integer(),
#>   state = col_character(),
#>   state_abbr = col_character(),
#>   house12 = col_double(),
#>   sen12 = col_double(),
#>   support_expansion_new = col_character(),
#>   percent_uninsured = col_double(),
#>   ideology = col_double(),
#>   income = col_double(),
#>   percent_black = col_double(),
#>   percent_hispanic = col_double(),
#>   percent_metro = col_double(),
#>   dsh = col_double(),
#>   obama_share = col_double()
#> )
#> See spec(...) for full column specifications.

br_scaled <- br %>%
  # Autoscale all vars but response
  mutate_at(vars(-oppose_expansion), autoscale)

glm(f, data = br, family = "binomial") %>% summary()
#> 
#> Call:
#> glm(formula = f, family = "binomial", data = br)
#> 
#> Deviance Residuals: 
#>    Min      1Q  Median      3Q     Max  
#> -2.374  -0.461  -0.131   0.630   2.207  
#> 
#> Coefficients:
#>                   Estimate Std. Error z value Pr(>|z|)   
#> (Intercept)         4.5103     4.5986    0.98    0.327   
#> dem_governor       -4.1556     1.4794   -2.81    0.005 **
#> obama_win          -2.1470     1.3429   -1.60    0.110   
#> gop_leg            -0.1865     1.2974   -0.14    0.886   
#> percent_uninsured  -0.3072     0.1651   -1.86    0.063 . 
#> income             -0.0421     0.0776   -0.54    0.587   
#> percent_nonwhite   17.8505    48.3030    0.37    0.712   
#> percent_metro     -12.4390    32.4446   -0.38    0.701   
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> (Dispersion parameter for binomial family taken to be 1)
#> 
#>     Null deviance: 68.593  on 49  degrees of freedom
#> Residual deviance: 37.948  on 42  degrees of freedom
#> AIC: 53.95
#> 
#> Number of Fisher Scoring iterations: 5

library("rstanarm")

fit1 <- stan_glm(f, data = br, family = "binomial")
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 7.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.76 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.227826 seconds (Warm-up)
#>                0.216907 seconds (Sampling)
#>                0.444733 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.2258 seconds (Warm-up)
#>                0.234074 seconds (Sampling)
#>                0.459874 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.232678 seconds (Warm-up)
#>                0.224351 seconds (Sampling)
#>                0.457029 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.221657 seconds (Warm-up)
#>                0.224032 seconds (Sampling)
#>                0.445689 seconds (Total)

fit2 <- stan_glm(f, data = br, prior = NULL, family = "binomial")
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 1).
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
#>  Elapsed Time: 1.56167 seconds (Warm-up)
#>                0.243438 seconds (Sampling)
#>                1.80511 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 2).
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
#>  Elapsed Time: 1.37833 seconds (Warm-up)
#>                0.208435 seconds (Sampling)
#>                1.58676 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 3).
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
#>  Elapsed Time: 1.06808 seconds (Warm-up)
#>                0.245541 seconds (Sampling)
#>                1.31363 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 4).
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
#>  Elapsed Time: 1.21401 seconds (Warm-up)
#>                0.217275 seconds (Sampling)
#>                1.43128 seconds (Total)
```




## Negative Binomial

The [Negative Binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution) model is also used for unbounded count data,
$$
Y = 0, 1, \dots, \infty
$$
The Poisson distribution has the restriction that the mean is equal to the variance, $\E(X) = \Var(X) = \lambda$.
The Negative Binomial distribution has an additional parameter that allows the variance to vary (though it is always larger than the mean).

The outcome is modeled as a negative binomial distribution,
$$
y_i \sim \dbinom(\alpha_i, \beta)
$$
with shape $\alpha \in \R^{+}$ and inverse scale $\beta \in \R^{+}$, and $\E(y) = \alpha_i / \beta$ and $\Var(Y) = \frac{\alpha_i}{\beta^2}(\beta + 1)$.
Then the mean can be modeled and transformed to the 
$$
\begin{aligned}[t]
\mu_i &= \log( \vec{x}_i \vec{\gamma} ) \\
\alpha_i &= \mu_i / \beta
\end{aligned}
$$


**Important** The negative binomial distribution has many different parameterizations.
An alternative parameterization of the negative binomial uses the mean and a over-dispersion parameter. 
$$
y_i \sim \dnbinomalt(\mu_i, \phi)
$$
with location parameter $\mu \in \R^{+}$ and over-dispersion parameter $\phi \in \R^{+}$, and $\E(y) = \mu_i$ and $\Var(Y) = \mu_i  + \frac{\mu_i^2}{\phi}$.
Then the mean can be modeled and transformed to the 
$$
\begin{aligned}[t]
\mu_i &= \log( \vec{x}_i \vec{\gamma} ) \\
\end{aligned}
$$

In Stan, there are multiple parameterizations of the 

- `neg_binomial_lpdf(y | alpha, beta)`with shape parameter `alpha` and inverse scale parameter `beta`.
- `neg_binomial_2_lpdf(y | mu, phi)` with mean `mu` and over-dispersion parameter `phi`.
- `neg_binomial_2_log_lpdf(y | eta, phi)` with log-mean `eta` and over-dispersion parameter `phi`

Also, `rstanarm` supports Poisson and [negative binomial models](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html).

- @BDA3 [Ch 16]

### References

For general references on count models see

- @GelmanHill2007a [p. 109-116]
- @McElreath2016a [Ch 10]
- @Fox2016a [Ch. 14]
- @BDA3 [Ch. 16]


## Multinomial / Categorical Models

## Gamma Regression

The response variable is continuous and positive. 
In gamma regression, the coefficient of variation is constant rather than the variance.
$$
y_i \sim \dgamma(\alpha_i, \beta)
$$
and 
$$
\begin{aligned}[t]
\alpha_i &= \mu_i / \beta \\
\mu_i &= \vec{x}_i \vec{\gamma}
\end{aligned}
$$

In Stan,

- `gamma(y | alpha, beta)` with shape parameter $\alpha > 0$ and inverse scale parameter $\beta > 0$. Then $\E(Y) = \alpha / \beta$ and $\Var(Y) = \alpha / \beta^2$.

## Beta Regression

This is for a response variable that is a proportion, $y_i \in (0, 1)$,
$$
y_i \sim \dbeta(\alpha_i, \beta_i)
$$
and
$$
\begin{aligned}[t]
\mu_i &= g^{-1}(\vec{x}_i' \vec{\gamma}) \\
\alpha_i &= \mu_i \phi \\
\beta_i &= (1 - \mu_i) \phi 
\end{aligned}
$$
Additionally, the $\phi$ parameter could also be modeled.

In Stan:

- `beta(y | alpha, beta)` with positive prior successes plus one, $\alpha > 0$, and negative prior failures plus one, $\beta > 0$. Then $\E(Y) = \alpha / (\alpha + \beta)$ and $\Var(Y) = \alpha\beta / ((\alpha + \beta)^2 (\alpha + \beta + 1))$.

**rstanarm** function [rstasnarm](https://www.rdocumentation.org/packages/rstasnarm/topics/stan_betareg)

See:

- @FerrariCribari-Neto2004a, @Cribari-NetoZeileis2010a, and @GruenKosmidisZeileis2012a on beta regression.
- **rstanarm** documentation [Modeling Rates/Proportions using Beta Regression with rstanarm](https://cran.r-project.org/web/packages/rstanarm/vignettes/betareg.html)


## Ordered Logistic

See **rstanarm** function [rstasnarm](https://www.rdocumentation.org/packages/rstasnarm/topics/stan_polr)

- @GelmanHill2007a [Ch 6.5]
- *rstanarm** vignette [Estimating Ordinal Regression Models with rstanarm](https://cran.r-project.org/web/packages/rstanarm/vignettes/polr.html)

## References

Texts:

- @BDA3 [Ch 16]
- @GelmanHill2007a [Ch. 5-6]
- @McElreath2016a [Ch. 9]
- @King1998a discusses MLE estimation of many common GLM models
- Many econometrics/statistics textbooks, e.g. @Fox2016a, discuss GLMs. Though
    they are not derived from a Bayesian context, they can easily transferred.
