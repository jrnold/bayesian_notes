
# Generalized Linear Models

## Prerequisites {-}


```r
library("rstan")
library("tidyverse")
```

## Introduction

Generalized linear models (GLMs) are a class of commonly used models.
In GLMs, the mean is specified as a function of a linear model of predictors,
$$
E(Y) = \mu = g^{-1}(\mat{X} \vec{\beta}) .
$$
GLMs are a generalization of linear regression from an unbounded continuous outcome variable to other types of data: binary, count, categorical, bounded continuous.

A GLM consists of three components:

1.  A *probability distribution* (*family*) specifying the conditional
    distribution of the response variable.
    In GLMs, the distribution is in the exponential family: Normal, Binomial, Poisson, Categorical, Multinomial, Poisson, Beta.

1.  A *linear predictor*, which is a linear function of the predictors,
    $$
    \eta = \mat{X} \vec{\beta}.
    $$

1.  A *link function* ($g(.)$) which maps the expected value to the  the linear predictor,
    $$
    g(\mu) = \eta .
    $$
    The link function is smooth and invertible, and the  *inverse link function* or *mean function* maps the linear predictor to the mean,
    $$
    \mu = g^{-1}(\eta) .
    $$
    The link function ($g$) and its inverse ($g^{-1}) translate $\eta$ from $(\-infty, +\infty)$ to the proper range for the probability distribution and back again.

These models are often estimated with MLE, as with the function [stats](https://www.rdocumentation.org/packages/stats/topics/glm).
These are also easily estimated in a Bayesian setting.

See the help for [stats](https://www.rdocumentation.org/packages/stats/topics/family) for common probability distributions, [stats](https://www.rdocumentation.org/packages/stats/topics/make.link) for common links,  and the [Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model) page for a table of common GLMs.
See the function **[VGAM](https://cran.r-project.org/package=VGAM)** for even more examples of link functions and probability distributions.

<!--
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
-->
<!--
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

-->

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

-   `poisson_lpdf`
-   `poisson_log_lpdf`: Poisson with a log link. This is for numeric stability.

Also, `rstanarm` supports the [Poisson](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html).

## Example

A regression model of bilateral sanctions for the period 1939 to 1983.
The outcome variable is the number of countries imposing sanctions.

```r
data("sanction", package = "Zelig")
```

TODO

## Negative Binomial

The [Negative Binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution) model is also used for unbounded count data,
$$
Y = 0, 1, \dots, \infty
$$
The Poisson distribution has the restriction that the mean is equal to the variance, $\E(X) = \Var(X) = \lambda$.
The Negative Binomial distribution has an additional parameter that allows the variance to vary (though it is always larger than the mean).

The outcome is modeled as a negative binomial distribution,
$$
y_i \sim \dBinom(\alpha_i, \beta)
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

-   `neg_binomial_lpdf(y | alpha, beta)`with shape parameter `alpha` and inverse scale parameter `beta`.
-   `neg_binomial_2_lpdf(y | mu, phi)` with mean `mu` and over-dispersion parameter `phi`.
-   `neg_binomial_2_log_lpdf(y | eta, phi)` with log-mean `eta` and over-dispersion parameter `phi`

Also, `rstanarm` supports Poisson and [negative binomial models](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html).

-   @BDA3 [Ch 16]

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

-   `gamma(y | alpha, beta)` with shape parameter $\alpha > 0$ and inverse scale parameter $\beta > 0$. Then $\E(Y) = \alpha / \beta$ and $\Var(Y) = \alpha / \beta^2$.

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

-   `beta(y | alpha, beta)` with positive prior successes plus one, $\alpha > 0$, and negative prior failures plus one, $\beta > 0$. Then $\E(Y) = \alpha / (\alpha + \beta)$ and $\Var(Y) = \alpha\beta / ((\alpha + \beta)^2 (\alpha + \beta + 1))$.

**rstanarm** function [rstasnarm](https://www.rdocumentation.org/packages/rstasnarm/topics/stan_betareg)

See:

-   @FerrariCribari-Neto2004a, @Cribari-NetoZeileis2010a, and @GruenKosmidisZeileis2012a on beta regression.
-   **rstanarm** documentation [Modeling Rates/Proportions using Beta Regression with rstanarm](https://cran.r-project.org/web/packages/rstanarm/vignettes/betareg.html)

## References

For general references on count models see

-   @GelmanHill2007a [p. 109-116]
-   @McElreath2016a [Ch 10]
-   @Fox2016a [Ch. 14]
-   @BDA3 [Ch. 16]

@BDA3 [Ch 16], @GelmanHill2007a [Ch. 5-6], @McElreath2016a [Ch. 9]. @King1998a discusses MLE estimation of many common GLM models.

Many econometrics/statistics textbooks, e.g. @Fox2016a, discuss GLMs. Though they are not derived from a Bayesian context, they can easily transferred.
