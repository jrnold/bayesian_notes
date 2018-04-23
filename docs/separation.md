
---
output: html_document
editor_options: 
  chunk_output_type: console
---

# Separation


```r
library("rstan")
library("rstanarm")
library("tidyverse")
library("recipes")
```

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

These are the priors suggested by [Stan](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) and 
used by default in **rstanarm** [rstanarm](https://www.rdocumentation.org/packages/rstanarm/topics/stan_glm).

## Example: Complete Separation Data

The following data is an example of data with complete separation.[^fake-separation]

```r
data1 <- tribble(
  ~y, ~x1, ~x2,
  0, 1, 3,
  0, 2, 2,
  0, 3, -1,
  0, 3, -1,
  1, 5, 2,
  1, 6, 4,
  1, 10, 1,
  1, 11, 0  
)
```


```r
count(data1, y, x1) %>%
  group_by(x1) %>%
  mutate(p = n / sum(n)) %>%
  select(-n) %>% 
  spread(y, p, fill = 0)
#> # A tibble: 7 x 3
#> # Groups:   x1 [7]
#>      x1   `0`   `1`
#>   <dbl> <dbl> <dbl>
#> 1    1.    1.    0.
#> 2    2.    1.    0.
#> 3    3.    1.    0.
#> 4    5.    0.    1.
#> 5    6.    0.    1.
#> 6   10.    0.    1.
#> # ... with 1 more row
```

The variable `x1` perfectly separates `y`, since when `x1 <= 3`, `y = 0`,
and when `x1 > 3`, `y = 1`.


```r
glm(y ~ x1 + x2, data = data1, family = binomial()) %>%
  summary()
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> 
#> Call:
#> glm(formula = y ~ x1 + x2, family = binomial(), data = data1)
#> 
#> Deviance Residuals: 
#>         1          2          3          4          5          6  
#> -2.10e-08  -1.40e-05  -2.52e-06  -2.52e-06   1.56e-05   2.10e-08  
#>         7          8  
#>  2.10e-08   2.10e-08  
#> 
#> Coefficients:
#>              Estimate Std. Error z value Pr(>|z|)
#> (Intercept)    -66.10  183471.72       0        1
#> x1              15.29   27362.84       0        1
#> x2               6.24   81543.72       0        1
#> 
#> (Dispersion parameter for binomial family taken to be 1)
#> 
#>     Null deviance: 1.1090e+01  on 7  degrees of freedom
#> Residual deviance: 4.5454e-10  on 5  degrees of freedom
#> AIC: 6
#> 
#> Number of Fisher Scoring iterations: 24
```

## Example: Quasi-Separation

The following generated data is an example of quasi-separation.[^quasi-separation]


```r
data2 <- tibble(
  y = c(0, 0, 0, 0, 1, 1, 1, 1, 1, 1),
  x1 = c(1, 2, 3, 3, 3, 4, 5, 6, 10, 11),
  x2 = c(3, 0, -1, 4, 1, 0, 2, 7, 3, 4)
)
```


```r
count(data2, y, x1) %>%
  group_by(x1) %>%
  mutate(p = n / sum(n)) %>%
  select(-n) %>% 
  spread(y, p, fill = 0)
#> # A tibble: 8 x 3
#> # Groups:   x1 [8]
#>      x1   `0`   `1`
#>   <dbl> <dbl> <dbl>
#> 1    1. 1.00  0.   
#> 2    2. 1.00  0.   
#> 3    3. 0.667 0.333
#> 4    4. 0.    1.00 
#> 5    5. 0.    1.00 
#> 6    6. 0.    1.00 
#> # ... with 2 more rows
```

The variable `x1` almost perfectly separates `y`.
When `x1 < 3`, then `y = 0`.
When `x1 > 3`, then `y = 1`.
Only when `x1 = 3`, does `y` takes values of either 0 or 1.


```r
glm(y ~ x1 + x2, data = data2, family = binomial()) %>%
  summary()
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> 
#> Call:
#> glm(formula = y ~ x1 + x2, family = binomial(), data = data2)
#> 
#> Deviance Residuals: 
#>     Min       1Q   Median       3Q      Max  
#> -1.0042  -0.0001   0.0000   0.0000   1.4689  
#> 
#> Coefficients:
#>              Estimate Std. Error z value Pr(>|z|)
#> (Intercept)   -58.076  17511.903     0.0     1.00
#> x1             19.178   5837.301     0.0     1.00
#> x2             -0.121      0.610    -0.2     0.84
#> 
#> (Dispersion parameter for binomial family taken to be 1)
#> 
#>     Null deviance: 13.4602  on 9  degrees of freedom
#> Residual deviance:  3.7792  on 7  degrees of freedom
#> AIC: 9.779
#> 
#> Number of Fisher Scoring iterations: 21
```

## Example: Support of ACA Medicaid Expansion

This example is from @Rainey2016a from the original paper @BarrilleauxRainey2014a
with replication code [here](https://github.com/carlislerainey/separation).
Load the data included in the **jrnold.bayes.notes** package:

```r
data("politics_and_need", package = "jrnold.bayes.notes")
```

What happens when estimated with GLM?

```r
glm(oppose_expansion ~ gop_governor + percent_favorable_aca + gop_leg +
         percent_uninsured + bal2012 + multiplier + percent_nonwhite +
         percent_metro,
       data = politics_and_need, family = binomial()) %>%
      summary()
#> 
#> Call:
#> glm(formula = oppose_expansion ~ gop_governor + percent_favorable_aca + 
#>     gop_leg + percent_uninsured + bal2012 + multiplier + percent_nonwhite + 
#>     percent_metro, family = binomial(), data = politics_and_need)
#> 
#> Deviance Residuals: 
#>    Min      1Q  Median      3Q     Max  
#> -1.738  -0.455   0.000   0.591   2.350  
#> 
#> Coefficients:
#>                        Estimate Std. Error z value Pr(>|z|)
#> (Intercept)           -1.94e+01   3.22e+03   -0.01     1.00
#> gop_governor           2.03e+01   3.22e+03    0.01     0.99
#> percent_favorable_aca  7.31e-03   8.88e-02    0.08     0.93
#> gop_leg                2.43e+00   1.48e+00    1.64     0.10
#> percent_uninsured      1.12e-01   2.72e-01    0.41     0.68
#> bal2012               -7.12e-04   1.14e-02   -0.06     0.95
#> multiplier            -3.22e-01   1.08e+00   -0.30     0.77
#> percent_nonwhite       4.52e-02   8.25e-02    0.55     0.58
#> percent_metro         -7.75e-02   4.74e-02   -1.64     0.10
#> 
#> (Dispersion parameter for binomial family taken to be 1)
#> 
#>     Null deviance: 62.687  on 49  degrees of freedom
#> Residual deviance: 31.710  on 41  degrees of freedom
#> AIC: 49.71
#> 
#> Number of Fisher Scoring iterations: 19
```

For Stan, preprocess the data:

```r
rec <- recipe(oppose_expansion ~ gop_governor + percent_favorable_aca + 
                gop_leg + percent_uninsured + bal2012 + multiplier +
                percent_nonwhite + percent_metro,
              data = politics_and_need) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep(politics_and_need, retain = TRUE)

X <- juice(rec, composition = "matrix")
y <- juice(rec, composition = "matrix")
```


Estimate with **rstanarm**.

```r
f <-oppose_expansion ~ gop_governor + percent_favorable_aca + 
      gop_leg + percent_uninsured + bal2012 + multiplier +
      percent_nonwhite + percent_metro
fit1 <- stan_glm(f, data = politics_and_need, family = "binomial")
```

What if no prior is used? Compare estimates and efficiency.

```r
fit2 <- stan_glm(f, data = politics_and_need, prior = NULL, family = "binomial")
```

## References

@Rainey2016a provides a mixed MLE/Bayesian simulation based approach to apply a prior to the variable with separation, while keeping the other coefficients at their MLE values.
Since the results are highly sensitive to the prior, multiple priors should be tried (informative, skeptical, and enthusiastic).

@Firth1993a suggests a data-driven Jeffreys invariant prior. This prior was also recommended in @Zorn2005a.

@GreenlandMansournia2015a suggest a log-F prior distribution which has an intuitive interpretation related to the number of observations.

[^fake-separation]: [FAQ: What is Complete or Quasi-Complete Separation in Logistic/Probit Regression and How do We Deal With Them?](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-complete-or-quasi-complete-separation-in-logisticprobit-regression-and-how-do-we-deal-with-them/)
