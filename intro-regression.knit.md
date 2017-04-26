
# Introduction to Stan and Linear Regression

This chapter is an introduction to writing and running a Stan model in R.
Also see the **rstan** [vignette](https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html) for similar content.


## Prerequites

For this section we will use the `duncan` dataset included in the **car** package.
Duncan's occupational prestige data is an example dataset used throughout the popular Fox regression text, *Applied Regression Analysis and Generalized Linear Models* [@Fox2016a].
It is originally from @Duncan1961a consists of survey data on the prestige of occupations in the US in 1950, and several predictors: type of occupation, income, and education of that 

```r
data("Duncan", package = "car")
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
y_i \sim N(\beta'x_i, \sigma^2)
$$

**Priors:** The model needs to specify a prior distribution for the parameters $(\beta, \sigma)$.
Rather than specify a single distribution for $\beta$ and $\sigma$, it will be easier to specify independent (separate) distributions for $\beta$ and $\sigma$.
The Stan manual and ... provide 
For the normal distribution, assume i.i.d. normal distributions for each element of $\beta$:
$$
\beta_k \sim N(b, s)
$$
For the scale parameter of the normal distribution, $\sigma$, we will use a half-Cauchy.
The Cauchy distribution is a special case of the Student t distribution when the degrees of freedom is 1.
In Bayesian stats, it has the property that it concentrates probability mass around its median (zero), but has very wide tails, so if the prior distribution guess is wrong, the parameter can still adapt to data.
A half-Cauchy distribution is a Cauchy distribution but with support of $(0, \infty)$ instead of the entire real line.
$$
\sigma \sim C^{+}(0, w)
$$

Combining all the previous equations, our statistical model for linear regression is,
$$
\begin{aligned}[t]
y &\sim N(\mu, \sigma) \\
\mu &= X \beta \\
\beta &\sim N(b, s) \\
\sigma &\sim C^{+}(0, w)
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
}
```


```r
mod1 <- stan_model("stan/lm.stan")
```















