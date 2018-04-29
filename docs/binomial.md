
---
output: html_document
editor_options:
  chunk_output_type: console
---
# Binomial Models

## Prerequisites {-}


```r
library("rstan")
library("rstanarm")
library("tidyverse")
library("recipes")
library("bayz")
```

## Introduction

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

## Link Functions {link-function}

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

Notes:

-   The logistic distribution is approximately a Student-t with df=7.

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

Estimate a model of vote turnout in the 1992 from the American National Election Survey (ANES) as a function of race, age, and education.
The data and example is from the Zelig library [Zelig](https://www.rdocumentation.org/packages/Zelig/topics/turnout).[^ex-logit]
You can load it with

```r
data("turnout", package = "ZeligData")
```

### Stan

A general Stan model for estimating logit models is:

<!--html_preserve--><pre class="stan">
<code>// bernoulli_logit_1.stan
data {
  // number of observations
  int<lower=0> N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on regression coefficients
  real<lower=0.> scale_alpha;
  vector<lower=0.>[K] scale_beta;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
}
transformed parameters {
  vector[N] eta;

  eta = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  // likelihood
  y ~ bernoulli_logit(eta);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = bernoulli_rng(inv_logit(eta[i]));
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta[i]);
  }
}</code>
</pre><!--/html_preserve-->


```r
data("turnout", package = "ZeligData")
```

Vote choice (`vote`) is modeled as a function of age, age-squared, income, and race.

Preprocess the data to create the design matrix, `X`, and the response `y` using the **recipes** package.
We will need to center and scale the design matrix.

```r
turnout <- mutate(turnout, white = as.numeric(race == "white"))
rec_turnout <- recipe(vote ~ income + age + white,
                      data = turnout) %>%
  step_poly(age, options = list(degree = 2)) %>%
  prep(data = turnout, retain = TRUE)
X <- juice(rec_turnout, all_predictors(), composition = "matrix")
y <- juice(rec_turnout, all_outcomes(), composition = "matrix") %>%
  drop()
```


```r
mod1_data <- list(
  X = X,
  N = nrow(X),
  K = ncol(X),
  y = y,
  scale_alpha = 10,
  scale_beta = 2.5 * apply(X, 2, sd),
  use_y_rep = FALSE,
  use_log_lik = TRUE
)
```



#### rstanarm

The **rstanarm** package can estimate binomial models using the function `stan_glm`:


```r
fit2 <- stan_glm(vote ~ income + age + white, data = turnout)
```

## References

For general references on binomial models see @Stan2016a [Sec. 8.5], @McElreath2016a [Ch 10], @GelmanHill2007a [Ch. 5; Sec 6.4-6.5], @Fox2016a [Ch. 14], and @BDA3 [Ch. 16].

[^ex-logit]: Example from [Zelig-logit](http://docs.zeligproject.org/en/latest/zelig-logit.html).
