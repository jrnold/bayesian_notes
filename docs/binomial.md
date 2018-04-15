
# Binomial Models


```r
library("rstan")
library("rstanarm")
library("tidyverse")
library("rubbish")
library("bayz")
```


Binomial models are used to an outcome that is a bounded integer,
$$
y_i \in 0, 1, 2, \dots, n .
$$
The outcome is distributed Binomial,
$$
\begin{aligned}[t]
y_i \sim \dbin \left(n_i, \pi \right)
\end{aligned}
$$

A *binary outcome* is a common special case,
$$
y_i \in \{0, 1\},
$$
and
$$
\begin{aligned}[t]
y_i &\sim \dbin \left(1, \pi \right) & \text{for all $i$} \\
\end{aligned}
$$

Depending on the [link function](#link-functions), these are  logit and probit models that appear in the literature.

### Link Functions {link-function}

The parameter $\pi \in (0, 1)$ is often modeled with a link function is and a linear predictor.
$$
\pi_i = g^{-1}(\vec{x}_i \vec{\beta})
$$

There are several common link functions, but they all have to map $R \to (0, 1)$.[^binomialcdf]

-   **Logit:** The logistic function,
    $$
    \pi_i = \logistic(x_i\T \beta) = \frac{1}{1 + \exp(- x_i\T\beta)} .
    $$
    Stan function `softmax`.
-   **Probit:** The CDF of the normal distribution.
    $$
    \pi_i = \Phi(x_i\T \beta)
    $$
    Stan function `normal_cdf`.
-   **cauchit**: The CDF of the Cauchy distribution. Stan function `cauchy_cdf`.
-   **cloglog**: The inverse of the conditional log-log function (cloglog) is
    $$
    \pi_i = 1 - \exp(-\exp(x_i\T \beta)) .
    $$
    Stan function `inv_cloglog`.

[^binomialcdf]: Since the cumulative distribution function of a distribution maps reals to $(0, 1)$, any CDF can be used as a link function.

Of these link functions, the probit has the narrowest tails (sensitivity to outliers), followed by the logit, and cauchit.
The [cloglog](https://en.wikipedia.org/wiki/Generalized_linear_model) function is different in that it is asymmetric.[^cloglog]
At zero its value is above 0.5, whereas the cauchit, logit, and probit links all equal 0.5 at 0,

```r
make.link("cloglog")$linkinv(0)
#> [1] 0.632
```

[^cloglog]: @BeckKatzTucker1998a show that the cloglog link function can be derived from a grouped duration model with binary response variables.


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

<img src="binomial_files/figure-html/unnamed-chunk-3-1.png" width="70%" style="display: block; margin: auto;" />

### Stan

In Stan, the Binomial distribution has two implementations:

-   `binomial_lpdf`
-   `binomial_logit_lpdf`.

The later implementation is for numeric stability.
Taking an exponential of a value can be numerically unstable, and `binomial_logit_lpdf` input is on the logit scale:
Whereas,
$$
y_i \sim \mathsf{binomial}(1 / (1 + \exp(x_i \beta)))
$$
the following is true,
$$
y_i \sim \mathsf{binomial\_logit}(x_i \beta)
$$


### Example: Vote Turnout

A general Stan model for estimating logit models is:


```r
mod1
```

prelist(class = "stan")list(list(name = "code", attribs = list(), children = list("// Logit Model\n//\n// y ~ Bernoulli(p)\n// p = a + X B\n// b0 \\sim cauchy(0, 10)\n// b \\sim cauchy(0, 2.5)\ndata {\n  // number of observations\n  int N;\n  // response\n  // vectors are only real numbers\n  // need to use an array\n  int<lower = 0, upper = 1> y[N];\n  // number of columns in the design matrix X\n  int K;\n  // design matrix X\n  // should not include an intercept\n  matrix [N, K] X;\n}\ntransformed data {\n  # default scales same as rstanarm\n  # assume data is centered and scaled\n  real<lower = 0.0> a_scale;\n  vector<lower = 0.0>[K] b_scale;\n  a_scale = 10.0;\n  b_scale = rep_vector(2.5, K);\n}\nparameters {\n  // regression coefficient vector\n  real a;\n  vector[K] b;\n}\ntransformed parameters {\n  vector<lower = 0.0, upper = 1.0>[N] p;\n  p = inv_logit(a + X * b);\n}\nmodel {\n  // priors\n  a ~ normal(0.0, a_scale);\n  b ~ normal(0.0, b_scale);\n  // likelihood\n  y ~ binomial(1, p);\n}\ngenerated quantities {\n  // simulate data from the posterior\n  vector[N] y_rep;\n  // log-likelihood posterior\n  vector[N] log_lik;\n  for (i in 1:N) {\n    y_rep[i] = binomial_rng(1, p[i]);\n    log_lik[i] = binomial_lpmf(y[i] | 1, p[i]);\n  }\n}")))

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

### Separation

Separation is when a predictor perfectly predicts a binary response variable [@Rainey2016a, @Zorn2005a].

-   *complete separation*: the predictor perfectly predicts both 0's and 1's.
-   *quasi-complete separation*: the predictor perfectly predicts either 0's or 1's.

This is related and similar to identification in MLE and multicollinearity in OLS.

The general solution is to penalize the likelihood, which in a Bayesian context is equivalent to placing a proper prior on the coefficient of the separating variable.

Using a weakly informative prior such as those suggested by is sufficient to solve separation,
$$
\beta_k \sim \dnorm(0, 2.5)
$$
where all the columns of $\code{x}$ are assumed to mean zero, unit variance (or otherwise standardized).
The half-Cauchy prior, $\dhalfcauchy(0, 2.5)$, suggested in @GelmanJakulinPittauEtAl2008a is insufficiently informative to  to deal with separation [@GhoshLiMitra2015a], but finite-variance weakly informative Student-t or Normal distributions will work.

These are the priors suggested by [Stan](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) and used by default in **rstanarm** [rstanarm](https://www.rdocumentation.org/packages/rstanarm/topics/stan_glm).

@Rainey2016a provides a mixed MLE/Bayesian simulation based approach to apply a prior to the variable with separation, while keeping the other coefficients at their MLE values.
Since the results are highly sensitive to the prior, multiple priors should be tried (informative, skeptical, and enthusiastic).

@Firth1993a suggests the Jeffreys invariant prior,
$$
p(\beta_k) \propto |I(\beta)|^{\frac{1}{2}}
$$
where $|I(\beta)|$ is the information matrix,
$$
\begin{aligned}[t]
I(\beta) &= \mat{X}\T \mat{W} \mat{X} \\
\mat{W} &= \diag(\pi_i (1 - \pi_i))
\end{aligned}
$$
This is the Jeffreys invariant prior. This was also recommended @Zorn2005a.

@GreenlandMansournia2015a suggest a log-F prior distribution which has an intuitive interpretation related to the number of observations.

#### Example: Support of ACA Medicaid Expansion

This example is from @Rainey2016a from the original paper @BarrilleauxRainey2014a
with replication code [here](https://github.com/carlislerainey/separation).


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

fit1 <- stan_glm(f, data = br, family = "binomial")
fit2 <- stan_glm(f, data = br, prior = NULL, family = "binomial")
```

## Rare Events Logit

## Case Control

In binary outcome variables, sometimes it is useful to sample on the dependent variable. 
For example, @KingZeng2001a and @KingZeng2001b discuss applications with respect to conflicts in international relations.
For most country-pairs, for most years, there is no conflict.
If some data are costly to gather, it may be cost efficient to get data for conflicts and then randomly select a smaller number of non-conflicts on which to gather data.
The sample will no longer be representative, but the estimates can be corrected.

The reason this works well, is that if there are very few 1's, additional 0's have little influence on the estimation (@KingZeng2001a).
This should hold more generally will unbalanced classes; in some sense, the amount of effective observations is not much more than the number in the lowest category.

@KingZeng2001a propose two corrections: 

1. Correcting the intercept (prior correction)
2. Weighting observations

The *prior correction* notes that
$$
\pi_i = \frac{1}{1 + \exp(-\mat{X} \vec{beta})}
$$

The unbalanced sample only affects the intercept. If $\hat\beta_0$ is the intercept from the MLE, the case-control corrected intercept $\tilde{\beta}$ is,
$$
\tilde{\vec{\beta}}_0^* = \hat{\vec{\beta}}_0 - \ln \left(\frac{1 - \tau}{\tau} \frac{\bar{y}}{1 - \bar{y}} \right)
$$
In an MLE setting, this can be applied after estimation, but used in any predicted values.
In a Bayesian setting, this correct should be applied within the model by adding the offset to the estimation.

In a Stan model, this could be implemented by directly incrementing these values

``` stan
data {
  int N;
  int y[N];
  real tau;
}
transformed data {
  real offset;
  real y_mean;
  y_mean = mean(y);
  offset = log((1 - tau) / tau * (y_mean) / (1 - y_mean));
}
parameters {
  real alpha0;
}
transformed parameters {
  real alpha;
  alpha <- alpha0 - offset;
}
```

If there was uncertainty about $\tau$, then $\tau$ could be modeled as a parameter.
It may also be okay to only correct the intercept in a generated quantities block? (not sure).

An alternative approach is to use a *weighted likelihood*:

- ones are weighted by $\tau / \bar{y}$
- zeros are weighted by $(1 - \tau) / \bar{1 - \bar{y}}$

The log likelihood would then be
$$
\ln L_w(\beta | y) = w_1 \sum_{Y_i = 1} \ln (\pi_i) + w_0 \sum_{Y_i = 0} \ln (1 - \pi_i)
$$

In Stan, this can be implemented by directly weighting the log-posterior contributions of each observation.
For example, something like this,

``` stan
if (y[i]) {
  target += w * binomial_lpdf(1, pi[i])
} else {
  target += (1 - w) * binomial_lpdf(1, pi[i])
}
```

See the example for [`Zelig-relogit`](http://docs.zeligproject.org/en/latest/zelig-relogit.html)

#### References

- @Firth1993a proposes a penalized likelihood approach using the Jeffreys invariant prior
- @KingZeng2001b and @KingZeng2001a apply an approach similar to the penalized likelihood approach for the similar problem of rare events
- @Zorn2005a also suggests using the Firth logistic regression to avoid perfect separation
- @Rainey2016a shows that Cauchy(0, 2.5) priors can be used
- @GreenlandMansournia2015a provide another default prior to for binomial models: log F(1,1) and log F(2, 2) priors. These have the nice property that they are interpretable as additional observations.

### References

For general references on binomial models see @Stan2016a [Sec. 8.5], @McElreath2016a [Ch 10], @GelmanHill2007a [Ch. 5; Sec 6.4-6.5], @Fox2016a [Ch. 14], and @BDA3 [Ch. 16].
