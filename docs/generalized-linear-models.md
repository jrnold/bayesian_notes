
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

## Binomial

- The outcomes $Y$ are non-negative integers: $0, 1, 2, \dots, n_i$.
- The total number, $n_i$, can vary by observation.
- Special case: $n_i = 1$ for all $i \in (1, 0)$: logit, probit models.

The outcome is distributed Binomial,
$$
\begin{aligned}[t]
y_i \sim \dbinom\left(n_i, \pi \right)
\end{aligned}
$$

The parameter $\pi \in [0, 1]$ is modeled with a link function and a linear predictor.

There are several common link functions, but they all have to map $R \to (0, 1)$.[^binomialcdf]

**Logit:** The logistic function,
    $$
    \pi_i = \logistic(x_i\T \beta) = \frac{1}{1 + \exp(- x_i\T\beta)} .
    $$
    Stan function `softmax`.
- **Probit:** The CDF of the normal distribution.
    $$
    \pi_i = \Phi(x_i\T \beta)
    $$
    Stan function `normal_cdf`.

- **cauchit**: The CDF of the Cauchy distribution. Stan function `cauchy_cdf`.
- **cloglog**: The inverse of the conditional log-log function (cloglog) is
    $$
    \pi_i = 1 - \exp(-\exp(x_i\T \beta)) .
    $$
    Stan function `inv_cloglog`.

[^binomialcdf]: Since a CDF maps reals to $(0, 1)$, any CDF can be used as a link function.

These link-functions are plotted below.
Of these link functions, the probit has the narrowest tails (sensitivity to outliers), followed by the logit, and cauchit.
The [cloglog](https://en.wikipedia.org/wiki/Generalized_linear_model#Complementary_log-log_.28cloglog.29) function is different in that it is asymmetric; at zero its value is above 0.5, whereas the cauchit, logit, and probit links all equal 0.5 at 0,

```r
make.link("cloglog")$linkinv(0)
#> [1] 0.632
```


```r
map(c("logit", "probit", "cauchit", "cloglog"),  make.link) %>%
map_df(
  function(link) {
    tibble(x = seq(-4, 4, length.out = 101),
           y = link$linkinv(x),
           link_name = link$name)
  }
) %>%
  ggplot(aes(x = x, y = y, colour = link_name)) +
  geom_line()
```

<img src="generalized-linear-models_files/figure-html/unnamed-chunk-3-1.png" width="70%" style="display: block; margin: auto;" />


In Stan, the Binomial distribution has two implementations:

- `binomial_lpdf`
- `binomial_logit_lpdf`: Poisson with a log link. This implementation is for numeric stability.


### Example: Vote Turnout

A general Stan model for estimating logit models is:


```r
mod1
```

<pre>
  <code class="stan">// Logit Model
//
// y ~ Bernoulli(p)
// p = a + X B
// b0 \sim cauchy(0, 10)
// b \sim cauchy(0, 2.5)
data {
  // number of observations
  int N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];
  // number of columns in the design matrix X
  int K;
  // design matrix X
  matrix [N, K] X;
}
parameters {
  // regression coefficient vector
  real b0;
  vector[K] b;
}
transformed parameters {
  vector<lower = 0.0, upper = 1.0>[N] p;
  p = inv_logit(b0 + X * b);
}
model {
  // priors
  b0 ~ cauchy(0.0, 10.0);
  b ~ cauchy(0.0, 2.5);
  // likelihood
  y ~ binomial(1, p);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = binomial_rng(1, p[i]);
    log_lik[i] = binomial_lpmf(y[i] | 1, p[i]);
  }
}</code>
</pre>
This uses the default semi-informative priors in Gelman 2008 ...

Estimate a model of vote turnout in the 1992 from the American National Election Survey (ANES).
The data is from [Zelig](https://www.rdocumentation.org/packages/Zelig/topics/turnout).[^ex-logit]

```r
data("turnout", package = "Zelig")
```
Vote choice (`vote`) is modeled as a function of age, income, and race.

```r
mod_formula <- vote ~ poly(age, 2) + income + educate + race - 1
```

```r
mod1_data <- lm_preprocess(mod_formula, data = turnout)
```


[^ex-logit]: Example from [Zelig-logit](http://docs.zeligproject.org/en/latest/zelig-logit.html).

### Extensions

- Separation: 
- Rare events:

### Perfect Separation

- @Firth1993a proposes a penalized likelihood approach using the Jeffreys invariant prior
- @KingZeng2001b and @KingZeng2001a apply an approach similar to the penalized likelihood approach for the similar problem of rare events
- @Zorn2005a also suggests using the Firth logistic regression to avoid perfect separation
- @Rainey2016a shows that Cauchy(0, 2.5) priors can be used
- @GreenlandMansournia2015a provide another default prior to for binomial models: log F(1,1) and log F(2, 2) priors. These have the nice property that they are interpretable as additional observations.

### References

- @Stan2016a [Sec. 8.5]
- @McElreath2016a [Ch 10]
- @GelmanHill2007a [Ch. 5; Sec 6.4-6.5]
- @Fox2016a [Ch. 14]
- @BDA3 [Ch. 16]



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


### References

- @GelmanHill2007a [p. 109-116]
- @McElreath2016a [Ch 10]
- @Fox2016a [Ch. 14]
- @BDA3 [Ch. 16]


## Negative Binomial

The [Negative Binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution) model is also used for unbounded count data,
$$
Y = 0, 1, \dots, \infty
$$
The Poisson distribution has the restriction that the mean is equal to the variance, $\E(X) = \Var(X) = \lambda$.
The Negative Binomial distribution has an additional parameter that allows the variance to vary (though it is always larger than the mean).

The outcome is modeled as a negative binomial distribution,
$$
y_i \sim \dnbinom(\alpha_i, \beta)
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

### Multinomial / Categorical Models

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



**rstanarm** function [rstasnarm](https://www.rdocumentation.org/packages/rstasnarm/topics/stan_polr)

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
