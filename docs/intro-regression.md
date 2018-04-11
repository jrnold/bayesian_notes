
# Introduction to Stan and Linear Regression

This chapter is an introduction to writing and running a Stan model in R.
Also see the **rstan** [vignette](https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html) for similar content.


## Prerequites

For this section we will use the `duncan` dataset included in the **car** package.
Duncan's occupational prestige data is an example dataset used throughout the popular Fox regression text, *Applied Regression Analysis and Generalized Linear Models* [@Fox2016a].
It is originally from @Duncan1961a consists of survey data on the prestige of occupations in the US in 1950, and several predictors: type of occupation, income, and education of that 

```r
data("Duncan", package = "carData")
```

## The Statistical Model

The first step in running a Stan model is defining the Bayesian statistical model that will be used for inference.

Let's run the regression of occupational prestige on the type of occupation, income, and education:
$$
\begin{multline}
y_i = \beta_0 + \beta_1 I(\mathtt{type} = \mathtt{"prof"}) + \beta_2 I(\mathtt{type} = \mathtt{"wc"}) \\
+ \beta_3 \mathtt{income} + \beta_4 \mathtt{education} + \epsilon_i
\end{multline}
$$

```r
duncan_lm <- lm(prestige ~ type + income + education,
   data = Duncan)
duncan_lm
#> 
#> Call:
#> lm(formula = prestige ~ type + income + education, data = Duncan)
#> 
#> Coefficients:
#> (Intercept)     typeprof       typewc       income    education  
#>      -0.185       16.658      -14.661        0.598        0.345
```


There are $n = 45$ observations in the dataset.
Let $y$ be a $n \times 1$ vector of the values of `prestige`.
Let $X$ be the $n \times k$ design matrix of the regression.
In this case, $k = 5$, 
$$
X = \begin{bmatrix}
1 & \mathtt{typeprof} & \mathtt{typewc} & \mathtt{income} & \mathtt{education}
\end{bmatrix}
$$

In OLS, we get the frequentist estimates of $\hat{\beta}$ by minimizing the squared errors,
$$
\hat{\beta}_{OLS} = \argmin_{\beta} \sum_{i = 1}^n (y_i - \beta' x_i)^2 = \argmin \sum_{i = 1}^n \hat{\epsilon}_i
$$
For valid inference we need to make assumptions about $\epsilon_i$, namely that they are uncorrelated with $X$, $\Cov(\epsilon, X) = 0$, and that they are i.i.d, $\Cov(\epsilon_i, \epsilon_j) = 0$, $\Var(\epsilon_i) = \sigma^2$ for all $i$.
However, no specific distributional form is or needs to be assumed for $\epsilon$ since CLT results show that, asymptotically, the sampling distribution of $\beta$ is distributed normal.
Additionally, although $\hat\sigma^2 = \sum_{i = 1}^n \epsilon_i / (n - k - 1)$ is a estimator of $\sigma^2$, standard errors of the standard error of the regression are not directly provided.

In Bayesian inference, our target is the posterior distribution of the parameters, $\beta$ and $\sigma$:  $p(\beta, \sigma^2 | y, X)$.
Since all uncertainty in Bayesian inference is provided via probability, we will need to explicitly provide parametric distributions for the likelihood and parameters.

$$
p(\beta, \sigma | y, X) \propto p(y | \beta, \sigma) p(\beta, \sigma)
$$

For a Bayesian linear regression model, we'll need to specify distributions for $p(y | \beta, \sigma)$ and $p(\beta, \sigma)$.

**Likelihood:** $p(y_i | x_i, \beta, \sigma)$ suppose that the observations are distributed independent normal:
$$
y_i \sim \dnorm(\beta'x_i, \sigma^2)
$$

**Priors:** The model needs to specify a prior distribution for the parameters $(\beta, \sigma)$.
Rather than specify a single distribution for $\beta$ and $\sigma$, it will be easier to specify independent (separate) distributions for $\beta$ and $\sigma$.
The Stan manual and ... provide 
For the normal distribution, assume i.i.d. normal distributions for each element of $\beta$:
$$
\beta_k \sim \dnorm(b, s)
$$
For the scale parameter of the normal distribution, $\sigma$, we will use a half-Cauchy.
The Cauchy distribution is a special case of the Student t distribution when the degrees of freedom is 1.
In Bayesian stats, it has the property that it concentrates probability mass around its median (zero), but has very wide tails, so if the prior distribution guess is wrong, the parameter can still adapt to data.
A half-Cauchy distribution is a Cauchy distribution but with support of $(0, \infty)$ instead of the entire real line.
$$
\sigma \sim \dhalfcauchy(0, w)
$$

Combining all the previous equations, our statistical model for linear regression is,
$$
\begin{aligned}[t]
y &\sim \dnorm(\mu, \sigma) \\
\mu &= X \beta \\
\beta &\sim \dnorm(b, s) \\
\sigma &\sim \dhalfcauchy(0, w)
\end{aligned}
$$
This defines a Bayesian model gives us
$$
p(\beta, \sigma | y, X, b, s, w) \propto p(y | X, \beta) p(\beta | b, s) p(\sigma | w)
$$
The targets of inference in this model are the two parameters: $\beta$ (regression coefficients), and  $\sigma$ (standard deviation of the regression).
This is conditional on the observed or assumed quantities, which including both the data $y$ (response) and $X$ (predictors), as well the values defining the prior distributions: $b$, $s$, and $w$.


Now that we've defined a statistical model, we can write it as a Stan model.

Stan models are written in its own domain-specific language that focuses on declaring the statistical model (parameters, variables, distributions) while leaving the details of the sampling algorithm to Stan.

A Stan model consists of *blocks* which contain declarations of variables and/or statements.
Each block has a specific purpose in the model.

```
functions {
    // OPTIONAL: user-defined functions
}
data {
    // read in data ...
}
transformed data {
    // Create new variables/auxiliary variables from the data
}
parameters {
    // Declare parameters that will be estimated
}
transformed parameters {
    // Create new variables/auxiliary variables from the parameters
}
model {
    // Declare your probability model: priors, hyperpriors & likelihood
}
generated quantities {
    // Declare any quantities other than simulated parameters to be generated
}
```

The file `lm.stan` is a Stan model for the linear regression model previously defined.

```
data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // beta prior
  real b_loc;
  real<lower = 0.0> b_scale;
  // sigma prior
  real sigma_scale;
}
parameters {
  // regression coefficient vector
  vector[k] b;
  // scale of the regression errors
  real<lower = 0.0> sigma;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[n] mu;
  mu = a + X * b;
}
model {
  // priors
  b ~ normal(b_loc, b_scale);
  sigma ~ cauchy(0, sigma_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  for (i in 1:n) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}
```


```r
mod1 <- stan_model("stan/lm.stan")
```

```r
mod1
```

<pre>
  <code class="stan">data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;
  // design matrix X
  matrix [n, k] X;
  // beta prior
  real b_loc;
  real<lower = 0.0> b_scale;
  // sigma prior
  real sigma_scale;
}
parameters {
  // regression coefficient vector
  vector[k] b;
  // scale of the regression errors
  real<lower = 0.0> sigma;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[n] mu;
  mu = X * b;
}
model {
  // priors
  b ~ normal(b_loc, b_scale);
  sigma ~ cauchy(0, sigma_scale);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[n] y_rep;
  for (i in 1:n) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}</code>
</pre>


See the [Stan Modeling Language User's Guide and Reference Manual](http://mc-stan.org/documentation/) for details of the Stan Language.


**Note**Since a Stan model compiles to C++ code, you may receive some warning messages such as
```
/Library/Frameworks/R.framework/Versions/3.3/Resources/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
    static void set_zero_all_adjoints() {
                ^
In file included from file1d4a4d50faa.cpp:8:
In file included from /Library/Frameworks/R.framework/Versions/3.3/Resources/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
```
As long as your model compiles, you can ignore these compiler warnings (On the other hard, warnings that occur during sampling should not be ignored).
If the Stan model does not give you a syntax error when parsing the model, it should compile to valid C++.[^bugs][^c-warnings]
See 

[bugs]: In the rare case that the Stan parser transpiles the Stan model to C++ but cannot compile the C++ code, it is a bug in Stan. Follow the [instructions](http://mc-stan.org/issues/) on how to inform the Stan developers about bugs. 
[c-warnings]: The extended installation instructions for [MacOS/Linux](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Mac-or-Linux) and [Windows](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Windows) have instructions for adding compiler options to the R [Makevars](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Using-Makevars) file.


### Sampling

In order to sample from the model, we need to at least give it the values for the data to use: `n`, `k`, `y`, `X`, and the data associated with the priors.


```r
mod1_data <- list(
  y = Duncan$prestige,
  n = nrow(Duncan)
)
```
The data types in Stan are all numeric (either integers or reals), but they include matrices and vectors. 
However, there is nothing like a data frame in Stan. 
Whereas in the R function `lm` we can provide a formula and a data set for where to look for objects, and the function will create the appropriate $X$ matrix for the regression, we will need to create that matrix ourselves---expanding categorical variables to indicator variables, and expanding interactions and other functions of the predictors.
However, we need to do that all manually.
The function [stats](https://www.rdocumentation.org/packages/stats/topics/model.matrix) is the workhorse function used in `lm` and many other R functions to convert a formula into the matrix used in estimation.

```r
X <- model.matrix(prestige ~ type + income + education, data = Duncan)
mod1_data$X <- X
mod1_data$k <- ncol(X)
```

We still need to provide the values for the prior distributions.
For specific values of the prior distributions, assume uninformative priors for `beta` by setting the mean to zero and the variances to large numbers.
$$
\beta_k \sim \dnorm(0, 1000)
$$


```r
mod1_data$b_loc <- 0
mod1_data$b_scale <- 1000
```

For prior of the regression scale parameter $\sigma$, use a half-Cauchy distribution with a large scale parameter, which is a good choice for the priors of scale parameters.
<!--
In this case, `prestige` has values between 0 and 100.
This is like a proportion (actually, it is a proportion x 100), so ignoring the covariates, the maximum variance of a distribution would be if `prestige = 50`, when the standard deviation would be $\sqrt{p * (1 - p)} = 50$. So a scale parameter of 50 is appropriate,
-->
$$
\sigma \sim \dhalfcauchy(0, 50)
$$

```r
mod1_data$sigma_scale <- 50
```

Now, sample from the posterior, using the function `sampling`:

```r
mod1_fit <- sampling(mod1, data = mod1_data)
```


### Convergence Diagnostics and Model Fit

- **Convergence Diagnostics:** Is this the posterior distribution that you were looking for? These don't directly say anything about how "good" the model is in terms representing the data, they are only evaluating how well the sampler is doing at sampling the posterior distribution of the given model. If there are problems with these, then the sample results do not represent the posterior distribution, and your inferences will be biased.

    - `mcse`: 
    - `n_eff`: 
    - `Rhat`
    - `divergences`
    
- **Model fit:** Is this statistical model appropriate for the data? Or better than other models?

    - Posterior predictive checks    
    - Information criteria:
    
        - WAIC
        - Leave-one-out Cross-Validation

