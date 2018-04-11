
# Unbounded Count Models

Unbounded count models are models in which the response is a natural number with no upper bound,
$$
y_i = 0, 1, \dots, \infty .
$$

The two distributions most commonly used to model this are 

- Poisson
- Negative Binomial

## Poisson 

The [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution):
$$
y_i \sim \dpois(\lambda_i) ,
$$



## Negative Binomial

The [Negative Binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution) model is also used for unbounded count data,
$$
Y = 0, 1, \dots, \infty
$$
The Poisson distribution has the restriction that the mean is equal to the variance, $\E(X) = \Var(X) = \lambda$.
The Negative Binomial distribution has an additional parameter that allows the variance to vary (though it is always larger than the mean).

The outcome is modeled as a negative binomial distribution,
$$
y_i \sim \dbin(\alpha_i, \beta)
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
y_i \sim \dnbinalt(\mu_i, \phi)
$$
with location parameter $\mu \in \R^{+}$ and over-dispersion parameter $\phi \in \R^{+}$, and $\E(y) = \mu_i$ and $\Var(Y) = \mu_i  + \frac{\mu_i^2}{\phi}$.
Then the mean can be modeled and transformed to the 
$$
\begin{aligned}[t]
\mu_i &= \log( \vec{x}_i \vec{\gamma} ) \\
\end{aligned}
$$

### Stan 

In Stan, there are multiple parameterizations of the 

- `neg_binomial_lpdf(y | alpha, beta)`with shape parameter `alpha` and inverse scale parameter `beta`.
- `neg_binomial_2_lpdf(y | mu, phi)` with mean `mu` and over-dispersion parameter `phi`.
- `neg_binomial_2_log_lpdf(y | eta, phi)` with log-mean `eta` and over-dispersion parameter `phi`

Also, `rstanarm` supports Poisson and [negative binomial models](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html).

### Example: Number of Number o




### References

For general references on count models see

- @BDA3 [Ch 16]
- @GelmanHill2007a [p. 109-116]
- @McElreath2016a [Ch 10]
- @Fox2016a [Ch. 14]
- @BDA3 [Ch. 16]


where $y_i \in 0, 1, 2, \dots$.

The parameter $\lambda_i \in (0, \infty)$ is both the mean and variance of the distribution.

### Link functions

In many applications, $\lambda$, is modeled as some function of covariates or other parameters.

Since $\lambda_i$ must be positive, the most common link function is the log,
$$
\log(\lambda_i) = \exp(\vec{x}_i' \vec{\beta})
$$
which has the inverse,
$$
\lambda_i = \log(\vec{x}_i \vec{\beta})
$$

### Stan

In Stan, the Poisson distribution has two implementations:

- `poisson_lpdf`
- `poisson_log_lpdf`

The Poisson with a log link. This is for numeric stability.

In **rstanarm** use [rstanarm](https://www.rdocumentation.org/packages/rstanarm/topics/stan_glm) for a Poisson GLM.

- **rstanarm** vignette on [counte models](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html).

## Example: Bilateral Sanctions

A regression model of bilateral sanctions for the period 1939 to 1983 [@Martin1992a].
The outcome variable is the number of countries imposing sanctions.


```r
mod_poisson1 <- stan_model("stan/poisson1.stan")
```


```r
data("sanction", package = "Zelig")

f <- num ~ coop + target -1L
reg_model_data <- lm_preprocess(f, data = sanction)

sanction_data <-
  list(X = autoscale(reg_model_data$X),
       y = reg_model_data$y)
sanction_data$N <- nrow(sanction_data$X)
sanction_data$K <- ncol(sanction_data$X)
```


```r
fit_sanction_pois <- sampling(mod_poisson1, data = sanction_data)
#> 
#> SAMPLING FOR MODEL 'poisson1' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.09828 seconds (Warm-up)
#>                0.095359 seconds (Sampling)
#>                0.193639 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'poisson1' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.101347 seconds (Warm-up)
#>                0.103472 seconds (Sampling)
#>                0.204819 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'poisson1' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.101324 seconds (Warm-up)
#>                0.107138 seconds (Sampling)
#>                0.208462 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'poisson1' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.10045 seconds (Warm-up)
#>                0.093848 seconds (Sampling)
#>                0.194298 seconds (Total)
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
y_i \sim \dnegbinom(\alpha_i, \beta)
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

### Example: Economic Sanctions II {ex-econ-sanctions-2}

Continuing the [economic sanctions example](#ex-econ-sanctions-2) of @Martin1992a.



```r
mod_negbin1 <- stan_model("stan/negbin1.stan")
```



```r
fit_sanction_nb <- sampling(mod_negbin1, data = sanction_data, control = list(adapt_delta  = 0.95))
#> 
#> SAMPLING FOR MODEL 'negbin1' NOW (CHAIN 1).
#> 
#> Gradient evaluation took 7.6e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.76 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> [1] "Error in sampler$call_sampler(args_list[[i]]) : "                                                                                                    
#> [2] "  Exception thrown at line 48: neg_binomial_2_rng: Random number that came from gamma distribution is 4.11693e+10, but must be less than 1.07374e+09"
#> error occurred during calling the sampler; sampling not done
```


```r
summary(fit_sanction_nb, par = c("a", "b", "phi"))$summary
#> Stan model 'negbin1' does not contain samples.
#> NULL
```

We can compare the 

```r
loo_sanction_pois <- loo(extract_log_lik(fit_sanction_pois, "log_lik"))
#> Warning: Some Pareto k diagnostic values are too high. See help('pareto-k-
#> diagnostic') for details.
# loo_sanction_nb <- loo(extract_log_lik(fit_sanction_nb, "log_lik"))
```


```r
# loo::compare(loo_sanction_pois, loo_sanction_nb)
```


### References

For general references on count models see @GelmanHill2007a [p. 109-116], @McElreath2016a [Ch 10], @Fox2016a [Ch. 14], and @BDA3 [Ch. 16].
