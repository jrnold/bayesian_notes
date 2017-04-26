
# Heteroskedasticity and Robust Regression

## Prerequisites

**[VGAM](https://cran.r-project.org/package=VGAM)** is needed for the Laplace distribution.

```r
library("VGAM")
```


## Linear Regression with Student t distributed errors


Like OLS, Bayesian linear regression with normally distributed errors is sensitive to outliers.
The normal distribution has narrow tail probabilities.

This plots the normal, Double Exponential (Laplace), and Student-t (df = 4) distributions all with mean 0 and scale 1, and the surprise ($- log(p)$) at each point.
Higher surprise is a lower log-likelihood.
Both the Student-t and Double Exponential distributions have surprise values well below the normal in the ranges (-6, 6). [^tailareas]
This means that outliers impose less of a penalty on the log-posterior models using these distributions, and the regression line would need to move less to incorporate those observations since the error distribution will not consider them as unusual.



```r
z <- seq(-6, 6, length.out = 100)
bind_rows(
  tibble(z = z,
         p = dnorm(z, 0, 1),
         distr = "Normal"),
  tibble(z = z,
         p = dt(z, 4),
         distr = "Student-t (df = 4)"),
  tibble(z = z,
         p = VGAM::dlaplace(z, 0, 1),
         distr = "Double Exponential")) %>%
  mutate(`-log(p)` = -log(p)) %>%
  ggplot(aes(x = z, y = `-log(p)`, colour = distr)) + 
  geom_line()
      
```

<img src="robust_files/figure-html/unnamed-chunk-3-1.png" width="70%" style="display: block; margin: auto;" />


[^tailareas]: The Double Exponential distribution still has a thinner tail than the Student-t at higher values.



```r
mod_t
```

prelist()list(list(name = "code", attribs = list(class = "stan"), children = list("data {\n  // number of observations\n  int n;\n  // response vector\n  vector[n] y;\n  // number of columns in the design matrix X\n  int k;\n  // design matrix X\n  matrix [n, k] X;\n  // beta prior\n  real b_loc;\n  real<lower = 0.0> b_scale;\n  // sigma prior\n  real sigma_scale;\n}\nparameters {\n  // regression coefficient vector\n  vector[k] b;\n  // scale of the regression errors\n  real<lower = 0.0> sigma;\n  real<lower = 1.0> nu;\n}\ntransformed parameters {\n  // mu is the observation fitted/predicted value\n  // also called yhat\n  vector[n] mu;\n  mu = X * b;\n}\nmodel {\n  // priors\n  b ~ normal(b_loc, b_scale);\n  sigma ~ cauchy(0, sigma_scale);\n  nu ~ gamma(2, 0.1);\n  // likelihood\n  y ~ student_t(nu, mu, sigma);\n}\ngenerated quantities {\n  // simulate data from the posterior\n  vector[n] y_rep;\n  // log-likelihood values\n  vector[n] log_lik;\n  for (i in 1:n) {\n    y_rep[i] = student_t_rng(nu, mu[i], sigma);\n    log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);\n  }\n\n}")))


```r
unionization <- read_tsv("data/western1995/unionization.tsv",
         col_types = cols(
              country = col_character(),
              union_density = col_double(),
              left_government = col_double(),
              labor_force_size = col_number(),
              econ_conc = col_double()
            ))
mod_data <- preprocess_lm(union_density ~ left_government + log(labor_force_size) + econ_conc, data = unionization)
                                   
mod_data <- within(mod_data, {
  b_loc <- 0
  b_scale <- 100
  sigma_scale <- sd(y)
})
```

The `max_treedepth` parameter needed to be increased because in some runs it was hitting the maximum tree depth.
This is likely due to the wide tails of the Student t distribution.

```r
mod_t_fit <- sampling(mod_t, data = mod_data, control = list(max_treedepth = 11))
#> 
#> SAMPLING FOR MODEL 'rlm' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.811644 seconds (Warm-up)
#>                0.646364 seconds (Sampling)
#>                1.45801 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 1
#>                                                                                          count
#> Exception thrown at line 35: student_t_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'rlm' NOW (CHAIN 2).
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
#>  Elapsed Time: 0.829618 seconds (Warm-up)
#>                0.838252 seconds (Sampling)
#>                1.66787 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 2
#>                                                                                          count
#> Exception thrown at line 35: student_t_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'rlm' NOW (CHAIN 3).
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
#>  Elapsed Time: 0.755767 seconds (Warm-up)
#>                0.710474 seconds (Sampling)
#>                1.46624 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 3
#>                                                                                     count
#> Exception thrown at line 35: student_t_lpdf: Scale parameter is 0, but must be > 0!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
#> 
#> SAMPLING FOR MODEL 'rlm' NOW (CHAIN 4).
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
#>  Elapsed Time: 0.897055 seconds (Warm-up)
#>                0.758835 seconds (Sampling)
#>                1.65589 seconds (Total)
#> The following numerical problems occurred the indicated number of times on chain 4
#>                                                                                          count
#> Exception thrown at line 35: student_t_lpdf: Scale parameter is inf, but must be finite!     1
#> When a numerical problem occurs, the Hamiltonian proposal gets rejected.
#> See http://mc-stan.org/misc/warnings.html#exception-hamiltonian-proposal-rejected
#> If the number in the 'count' column is small, there is no need to ask about this message on stan-users.
```


```r
summary(mod_t_fit, pars = c("nu", "sigma", "b"))$summary
#>         mean se_mean      sd    2.5%    25%    50%     75%   97.5% n_eff
#> nu    21.699 0.24416 14.3626   3.599 11.279 18.431  28.918  57.369  3460
#> sigma 10.441 0.04339  2.0681   7.059  9.015 10.166  11.610  15.230  2272
#> b[1]  66.279 1.47339 53.0047 -43.640 32.578 67.960 102.590 166.056  1294
#> b[2]   0.274 0.00149  0.0806   0.114  0.223  0.275   0.326   0.432  2914
#> b[3]  -4.494 0.09324  3.4316 -11.013 -6.858 -4.598  -2.252   2.676  1354
#> b[4]  10.789 0.50310 18.3043 -23.366 -2.307 10.319  22.500  48.544  1324
#>        Rhat
#> nu    1.000
#> sigma 1.001
#> b[1]  1.002
#> b[2]  0.999
#> b[3]  1.002
#> b[4]  1.002
```

Compare those results when using a model with 


```r
mod_normal
```

prelist()list(list(name = "code", attribs = list(class = "stan"), children = list("data {\n  // number of observations\n  int n;\n  // response vector\n  vector[n] y;\n  // number of columns in the design matrix X\n  int k;\n  // design matrix X\n  matrix [n, k] X;\n  // beta prior\n  real b_loc;\n  real<lower = 0.0> b_scale;\n  // sigma prior\n  real sigma_scale;\n}\nparameters {\n  // regression coefficient vector\n  vector[k] b;\n  // scale of the regression errors\n  real<lower = 0.0> sigma;\n}\ntransformed parameters {\n  // mu is the observation fitted/predicted value\n  // also called yhat\n  vector[n] mu;\n  mu = X * b;\n}\nmodel {\n  // priors\n  b ~ normal(b_loc, b_scale);\n  sigma ~ cauchy(0, sigma_scale);\n  // likelihood\n  y ~ normal(mu, sigma);\n}\ngenerated quantities {\n  // simulate data from the posterior\n  vector[n] y_rep;\n  // log-likelihood posterior\n  vector[n] loglik;\n  for (i in 1:n) {\n    y_rep[i] = normal_rng(mu[i], sigma);\n    loglik[i] = normal_lpdf(y[i] | mu[i], sigma);\n  }\n}")))


```r
mod_normal_fit <- sampling(mod_normal, data = mod_data)
#> 
#> SAMPLING FOR MODEL 'lm' NOW (CHAIN 1).
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
#>  Elapsed Time: 0.483015 seconds (Warm-up)
#>                0.435345 seconds (Sampling)
#>                0.91836 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm' NOW (CHAIN 2).
#> 
#> Gradient evaluation took 9e-06 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.09 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.458847 seconds (Warm-up)
#>                0.349973 seconds (Sampling)
#>                0.80882 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm' NOW (CHAIN 3).
#> 
#> Gradient evaluation took 1.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.11 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.465702 seconds (Warm-up)
#>                0.335748 seconds (Sampling)
#>                0.80145 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'lm' NOW (CHAIN 4).
#> 
#> Gradient evaluation took 1.1e-05 seconds
#> 1000 transitions using 10 leapfrog steps per transition would take 0.11 seconds.
#> Adjust your expectations accordingly!
#> 
#> 
#> Iteration:    1 / 2000 [  0%]  (Warmup)
#> Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Iteration: 2000 / 2000 [100%]  (Sampling)
#> 
#>  Elapsed Time: 0.54117 seconds (Warm-up)
#>                0.407283 seconds (Sampling)
#>                0.948453 seconds (Total)
```


```r
summary(mod_normal_fit, pars = c("b", "sigma"))$summary
#>         mean se_mean      sd    2.5%    25%    50%     75%   97.5% n_eff
#> b[1]  72.930  1.5774 52.9423 -32.488 38.804 72.844 108.590 175.722  1127
#> b[2]   0.269  0.0019  0.0813   0.105  0.217  0.268   0.321   0.431  1835
#> b[3]  -4.859  0.1048  3.5445 -11.707 -7.235 -4.898  -2.621   2.184  1143
#> b[4]   8.368  0.5112 17.6960 -25.508 -3.307  8.001  20.290  43.378  1198
#> sigma 11.070  0.0621  2.1395   7.879  9.582 10.739  12.195  16.245  1188
#>       Rhat
#> b[1]     1
#> b[2]     1
#> b[3]     1
#> b[4]     1
#> sigma    1
```

Alternatively, the Double Exponential (Laplace) distribution can be used for the errors.
This is the equivalent to least quantile regression, where the regression line is the median (50% quantile)


```r
mod_dbl_exp
```

prelist()list(list(name = "code", attribs = list(class = "stan"), children = list("data {\n  // number of observations\n  int n;\n  // response vector\n  vector[n] y;\n  // number of columns in the design matrix X\n  int k;\n  // design matrix X\n  matrix [n, k] X;\n  // beta prior\n  real b_loc;\n  real<lower = 0.0> b_scale;\n  // sigma prior\n  real sigma_scale;\n}\nparameters {\n  // regression coefficient vector\n  vector[k] b;\n  // scale of the regression errors\n  real<lower = 0.0> sigma;\n}\ntransformed parameters {\n  // mu is the observation fitted/predicted value\n  vector[n] mu;\n  // tau are obs-level scale params\n  mu = X * b;\n}\nmodel {\n  // priors\n  b ~ normal(b_loc, b_scale);\n  sigma ~ cauchy(0, sigma_scale);\n  // likelihood\n  y ~ double_exponential(mu, sigma);\n}\ngenerated quantities {\n  // simulate data from the posterior\n  vector[n] y_rep;\n  // log-likelihood values\n  vector[n] log_lik;\n  // use a single loop since both y_rep and log_lik are elementwise\n  for (i in 1:n) {\n    y_rep[i] = double_exponential_rng(mu[i], sigma);\n    log_lik[i] = double_exponential_lpdf(y[i] | mu[i], sigma);\n  }\n\n}")))



```r
summary(mod_dbl_exp_fit, par = c("b", "sigma"))$summary
#>         mean se_mean      sd    2.5%    25%   50%    75%   97.5% n_eff
#> b[1]  38.693 1.90247 51.2313 -60.417  5.004 37.97 71.661 140.290   725
#> b[2]   0.298 0.00225  0.0837   0.131  0.245  0.30  0.352   0.458  1387
#> b[3]  -2.971 0.11773  3.2419  -9.408 -5.073 -3.01 -0.842   3.242   758
#> b[4]  20.981 0.67250 18.2690 -15.589  9.157 21.53 33.057  56.774   738
#> sigma  9.050 0.06676  2.2260   5.585  7.533  8.75 10.209  14.423  1112
#>       Rhat
#> b[1]     1
#> b[2]     1
#> b[3]     1
#> b[4]     1
#> sigma    1
```



## Heteroskedasticity

In applied regression, heteroskedasticity consistent (HC) or robust standard errors are often used.

However, there is straightforwardly direct translation of HC standard error to regression model this in a Bayesian setting. The sandwich method of estimating HC errors uses the same point estimates for the regression coefficients as OLS, but estimates the standard errors of those coefficients in a second stage from the OLS residuals. 
Disregarding differences in frequentist vs. Bayesian inference, it is clear that a direct translation of that method could not be fully Bayesian since the coefficients and errors are not estimated jointly.

In a linear normal regression model with heteroskedasticity, each observation has its own scale parameter, $\sigma_i$,
$$
\begin{aligned}[t]
y_i &\sim \dnorm(X \beta, \sigma_i) .
\end{aligned}
$$
It should be clear that without proper priors this model is not identified, meaning that the posterior distribution is improper.
To estimate this model we have to apply some model to the scale terms, $\sigma_i$.
In fact, you can think of homoskedasticity as the simplest such model; assuming that all $\sigma_i = \sigma$.
A more general model of $\sigma_i$ should encode any information the analyst has about the scale terms.
This can be a distribution or functions of covariates for how we think observations may have different values.

### Covariates

A simple model of heteroskedasticity is if the observations can be split into groups. Suppose the observations are partitioned into $k = 1, \dots, K$ groups, and $k[i]$ is the group of observation $i$,
$$
\sigma_i = \sigma_{k[i]}
$$

Another choice would be to model the scale term with a regression model, for example,
$$
\log(\sigma_i) \sim \dnorm(X \gamma, \tau)
$$


### Student-t

It turns out that the Student-t distribution of error terms from the [Robust Regression] chapter can also be derived as a model of heteroskedasticity.

A reparameterization that will be used quite often is to rewrite a normal distributions with unequal
scale parameters as a continuous mixture of a common global scale parameter ($\sigma$), and observation specific local scale parameters, $\lambda_i$,[^globalmixture]
$$
y_i \sim \dnorm(X\beta, \lambda_i \sigma) .
$$

If the local scale parameters are distributed as,
$$
\lambda^2 \sim \dinvgamma(\nu / 2, \nu / 2)
$$
then the above is equivalent to a regression with errors distributed Student-t errors with $\nu$ degrees of freedom,
$$
y_i \sim \dt{\nu}(X \beta, \sigma) .
$$

[^globalmixture] See [this](http://www.sumsar.net/blog/2013/12/t-as-a-mixture-of-normals/) for a visualization of a Student-t distribution a mixture of Normal distributions, and [this](https://www.johndcook.com/t_normal_mixture.pdf) for a derivation of the Student t distribution as a mixture of normal distributions. This scale mixture of normal representation will also be used with shrinkage priors on the regression coefficients.


**Example:** Simulate Student-t distribution with $\nu$ degrees of freedom as a scale mixture of normal. For *s in 1:S$,

1. Simulate $z_s \sim \dgamma(\nu / 2, \nu / 2)$
2. $x_s = 1 / \sqrt{z_s}2$ is draw from $\dt{\nu}(0, 1)$.

When using R, ensure that you are using the correct parameterization of the gamma distribution. **Left to reader**


## References

### Robust regression 

- See @GelmanHill2007a [sec 6.6], @BDA3 [ch 17]
- @Stan2016a [Sec 8.4] for the Stan example using a Student-t distribution

### Heteroskedasticity

- @BDA3 [Sec. 14.7] for models with unequal variances and correlations.
- @Stan2016a reparameterizes the Student t distribution as a mixture of gamma distributions in Stan.
