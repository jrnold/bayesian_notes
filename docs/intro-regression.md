
---
output: html_document
editor_options: 
  chunk_output_type: console
---
# Introduction to Stan and Linear Regression

This chapter is an introduction to writing and running a Stan model in R.
Also see the **rstan**
[vignette](https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html)
for similar content.

## Prerequisites {-}


```r
library("rstan")
library("tidyverse")
library("recipes")
```

For this section we will use the `duncan` dataset included in the **carData** package.
Duncan's occupational prestige data is an example dataset used throughout the popular Fox regression text, *Applied Regression Analysis and Generalized Linear Models* [@Fox2016a].
It is originally from @Duncan1961a consists of survey data on the prestige of occupations in the US in 1950, and several predictors: type of occupation, income, and education of that

```r
data("Duncan", package = "carData")
```

## OLS and MLE Linear Regression

The first step in running a Stan model is defining the Bayesian statistical model that will be used for inference.

We will model `prestige` of each occupation as a function of its `education`, `occupation`, and `type`.

A standard way to do this is with the OLS estimator:
$$
\begin{multline}
y_i = \beta_0 + \beta_1 I(\mathtt{type} = \mathtt{"prof"}) + \beta_2 I(\mathtt{type} = \mathtt{"wc"}) \\
\quad + \beta_3 \mathtt{income} + \beta_4 \mathtt{education} + \epsilon_i
\end{multline}
$$


```r
duncan_lm <- lm(prestige ~ type + income + education, data = Duncan)
```

$$
y_i = x_i' \beta + \epsilon_i
$$
OLS finds $\hat{\beta}_{OLS}$ by minimizing the squared errors,
$$
\hat{\beta}_{\text{OLS}} = \arg \min_{b} \sum_{i = 1}^n (y_i - x_i' b)^2 .
$$
OLS is an estimator of the (linear approximation of) the conditional expectation function,
$$
\mathrm{CEF}(y_i | x_i) = E(y_i, x_i' \beta) .
$$

For valid inference we need to make assumptions about $\epsilon_i$, namely that they are uncorrelated with $X$, $\Cov(\epsilon, X) = 0$, and that they are i.i.d, $\Cov(\epsilon_i, \epsilon_j) = 0$, $\Var(\epsilon_i) = \sigma^2$ for all $i$.
However, no specific distributional form is or needs to be assumed for $\epsilon$ since CLT results show that, asymptotically the sampling distribution of $\beta$ approaches the normal.
Additionally, although $\hat\sigma^2 = \sum_{i = 1}^n \epsilon_i / (n - k - 1)$ is a estimator of $\sigma^2$, standard errors of the standard error of the regression are not directly provided.

However, the OLS estimator is also the same as the MLE estimator for $\beta$ (but not $\sigma$):
$$
\begin{aligned}[t]
p(y_1, \dots, y_n | \beta, \sigma, x_1, \dots, x_n) &= \prod_{i = 1}^n p(y_i | \beta, x_i) \\
&= \prod_{i = 1}^n N(y_i | x_i' \beta) \\
&= \prod_{i = 1}^n \frac{1}{\sigma \sqrt{2 \pi}} \left( \frac{-(y_i - x_i' \beta)}{2 \sigma^2} \right)
\end{aligned}
$$
so,
$$
\hat{\beta}_{MLE}, \hat{\sigma}_{MLE} = \arg\max_{b,s} \prod_{i = 1}^n N(y_i | x_i' b, s^2)  .
$$
And $\hat{\beta}_{MLE} = \hat{\beta}_{OLS}$.

Note that the OLS estimator is equivalent to the MLE estimator of $\beta$,
$$
\begin{aligned}[t]
\hat{\beta}_{MLE} &= \arg \max_{b} \prod_{i = 1}^n N(y_i | x_i' b, \sigma^2) \\
&=  \arg \max_{b} \prod_{i = 1}^n \frac{1}{\sigma \sqrt{2 \pi}} \exp \left( \frac{-(y_i - x_i' \beta)^2}{2 \sigma^2} \right) \\
&= \arg \max_{b} \log \left( \prod_{i = 1}^n \frac{1}{\sigma \sqrt{2 \pi}} \exp \left( \frac{-(y_i - x_i' \beta)}{2 \sigma^2} \right) \right) \\
&= \arg \max_{b} \sum_{i = 1}^n - \log \sigma - \frac{1}{2} \log 2 \pi + \frac{-(y_i - x_i' \beta)^2}{2 \sigma^2} \\
&= \arg \max_{b} \sum_{i = 1}^n  -(y_i - x_i' \beta)^2 \\
&= \arg \min_{b} \sum_{i = 1}^n  (y_i - x_i' \beta)^2  \\
&= \hat{\beta}_{OLS}
\end{aligned}
$$
However, the estimator of $\sigma^2_{MLE} \neq \sigma^2_{OLS}$.

### Bayesian Model with Improper priors

In Bayesian inference, our target is the posterior distribution of the parameters, $\beta$ and $\sigma$:  $p(\beta, \sigma^2 | y, X)$.

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

We will use what are called an *improper uniform priors*.
An improper prior is,
$$
p(\theta) \propto C
$$
where $C$ is some constants.
This function puts an equal density on all values of the support of $\theta$.
This function is not a proper probability density function since $\int_{\theta \in \Theta} C d \theta = \infty$.
However, for some Bayesian models, the prior does not need to be a proper probability function for the posterior to be a probability function.
In this example we will put improper prior distributions on $\beta$ and $\sigma$.
$$
p(\beta, \sigma) = C
$$

$$
\begin{aligned}
p(\beta, \sigma | x, y) &\propto p(y| \beta, \sigma, x) p(\beta, \sigma, x) \\
&= \prod_{i = 1}^n N(y_i | x_i' \beta, \sigma^2) \cdot C \\
&\propto \prod_{i = 1}^n N(y_i | x_i' \beta, \sigma^2)
\end{aligned}
$$

Note that under the improper priors, the posterior is proportional to the likelihood,
$$
p(\beta, \sigma | x, y) \propto p(y | x, \beta, \sigma)
$$
Thus the MAP (maximum a posterior) estimator is the same as the MLE,
$$
\hat{\beta}_{MAP}, \hat{\sigma}_{MAP} = \arg\max_{\beta, \sigma} p(\beta, \sigma | x, y) = \arg \max_{\beta, \sigma} p(y | x, \beta, \sigma) = \hat{\beta}_{MLE}, \hat{\sigma}_{MLE}
$$

## Stan Model

Let's write and estimate our model in Stan.
Stan models are written in its own domain-specific language that focuses on declaring the statistical model (parameters, variables, distributions) while leaving the details of the sampling algorithm to Stan.

A Stan model consists of *blocks* which contain declarations of variables and/or statements.
Each block has a specific purpose in the model.

## Sampling Model with Stan

``` stan
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

The file `lm0.stan` is a Stan model for the linear regression model previously defined.

<!--html_preserve--><pre class="stan">
<code>// lm_normal_1.stan
// Linear Model with Normal Errors
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  vector<lower=0.>[K] scale_beta;
  real<lower=0.> loc_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  real<lower=0.> sigma;
}
transformed parameters {
  vector[N] mu;

  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  sigma ~ exponential(loc_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}</code>
</pre><!--/html_preserve-->


```r
mod1 <- stan_model("stan/lm_normal_1.stan")
```

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

In order to sample from the model, we need to at least give it the values for the data to use: ``, `k`, `y`, `X`, and the data associated with the priors.

The data types in Stan are all numeric (either integers or reals), but they
include matrices and vectors. However, there is nothing like a data frame in
Stan. Whereas in the R function `lm` we can provide a formula and a data set
for where to look for objects, and the function will create the appropriate $X$
matrix for the regression, we will need to create that matrix
ourselves---expanding categorical variables to indicator variables, and
expanding interactions and other functions of the predictors.


```r
rec <- recipe(prestige ~ income + education + type, data = Duncan) %>%
  step_dummy(type) %>%
  prep(data = Duncan, retain = TRUE) 
X <- juice(rec, all_predictors(), composition = "matrix")
y <- drop(juice(rec, all_outcomes(), composition = "matrix"))
```


```r
mod1_data <- list(
  X = X,
  K = ncol(X),
  N = nrow(X),
  y = y,
  use_y_rep = FALSE,
  use_log_lik = FALSE
)
```

We still need to provide the values for the prior distributions.
For specific values of the prior distributions, assume uninformative priors for `beta` by setting the mean to zero and the variances to large numbers.


```r
mod1_data$scale_alpha <- sd(y) * 10
mod1_data$scale_beta <- apply(X, 2, sd) * sd(y) * 2.5
mod1_data$loc_sigma <- sd(y)
```

Now, sample from the posterior, using the function `sampling`:

```r
mod1_fit <- sampling(mod1, data = mod1_data)
```


```r
summary(mod1_fit)
#> $summary
#>             mean  se_mean     sd     2.5%      25%      50%      75%
#> alpha     -0.249 0.036791 1.7422   -3.688   -1.452   -0.252    0.940
#> beta[1]    0.598 0.000843 0.0422    0.515    0.569    0.597    0.628
#> beta[2]    0.346 0.001233 0.0534    0.243    0.310    0.347    0.382
#> beta[3]   16.611 0.075172 3.2519   10.426   14.343   16.633   18.818
#> beta[4]  -14.567 0.059498 2.8363  -20.022  -16.467  -14.560  -12.681
#> sigma      4.569 0.004019 0.2087    4.168    4.428    4.561    4.706
#> mu[1]     83.207 0.017079 1.0801   81.060   82.506   83.230   83.902
#> mu[2]     85.722 0.020394 1.2438   83.209   84.894   85.741   86.532
#> mu[3]     93.056 0.022266 1.2739   90.549   92.218   93.074   93.883
#> mu[4]     80.408 0.018851 1.1923   78.056   79.647   80.415   81.197
#> mu[5]     84.403 0.017152 1.0848   82.236   83.700   84.420   85.094
#> mu[6]     58.006 0.036938 1.9884   54.086   56.631   58.023   59.369
#> mu[7]     86.827 0.020598 1.2088   84.468   86.023   86.835   87.614
#> mu[8]     98.815 0.030008 1.5203   95.783   97.793   98.823   99.839
#> mu[9]     55.364 0.035656 2.2551   50.985   53.853   55.320   56.856
#> mu[10]    89.185 0.018434 1.1658   86.880   88.405   89.215   89.950
#> mu[11]    67.096 0.022881 1.2851   64.629   66.208   67.095   67.954
#> mu[12]    95.732 0.026991 1.4133   92.916   94.764   95.732   96.683
#> mu[13]    95.385 0.026176 1.3884   92.604   94.440   95.391   96.311
#> mu[14]    69.961 0.022823 1.3524   67.311   69.054   69.972   70.856
#> mu[15]    76.570 0.022464 1.3341   74.003   75.688   76.569   77.461
#> mu[16]    42.389 0.051593 2.7231   36.996   40.566   42.395   44.286
#> mu[17]    63.628 0.047609 2.1508   59.454   62.188   63.662   65.076
#> mu[18]    71.622 0.035577 1.7183   68.240   70.487   71.639   72.781
#> mu[19]    56.706 0.049320 2.1788   52.484   55.207   56.729   58.155
#> mu[20]    91.386 0.021486 1.2981   88.875   90.498   91.411   92.230
#> mu[21]    27.454 0.034915 2.2082   23.001   25.980   27.454   28.943
#> mu[22]    32.924 0.029232 1.8488   29.263   31.714   32.917   34.144
#> mu[23]    42.650 0.029833 1.8868   38.934   41.400   42.651   43.895
#> mu[24]    19.835 0.032524 2.0570   15.834   18.428   19.827   21.196
#> mu[25]    20.269 0.016000 1.0119   18.350   19.563   20.266   20.950
#> mu[26]    41.353 0.028318 1.4324   38.527   40.378   41.357   42.303
#> mu[27]    57.867 0.050996 2.5800   52.801   56.069   57.819   59.663
#> mu[28]    32.353 0.019296 1.1310   30.120   31.610   32.341   33.098
#> mu[29]    20.521 0.016079 1.0169   18.581   19.816   20.510   21.212
#> mu[30]    34.711 0.022190 1.3210   32.196   33.806   34.711   35.629
#> mu[31]    18.761 0.018306 1.1024   16.642   18.008   18.737   19.509
#> mu[32]     6.359 0.028275 1.4339    3.523    5.362    6.361    7.340
#> mu[33]    33.861 0.020852 1.2577   31.419   33.018   33.862   34.729
#> mu[34]    11.711 0.020253 1.1748    9.478   10.868   11.702   12.500
#> mu[35]    17.499 0.018971 1.1363   15.259   16.727   17.484   18.283
#> mu[36]    19.230 0.016443 1.0399   17.249   18.520   19.224   19.943
#> mu[37]    18.320 0.016698 1.0561   16.293   17.604   18.299   19.036
#> mu[38]    19.012 0.017476 1.0740   16.937   18.276   18.983   19.740
#> mu[39]    11.018 0.020919 1.1939    8.742   10.173   11.018   11.829
#> mu[40]    15.739 0.017428 1.0772   13.705   14.982   15.727   16.479
#> mu[41]    17.314 0.020183 1.1740   15.053   16.506   17.305   18.096
#> mu[42]    18.571 0.016428 1.0390   16.581   17.854   18.556   19.271
#> mu[43]    10.862 0.021082 1.2097    8.570    9.998   10.846   11.674
#> mu[44]    36.352 0.031209 1.4842   33.485   35.353   36.354   37.313
#> mu[45]    15.615 0.023464 1.3057   13.125   14.715   15.594   16.460
#> lp__    -305.014 0.042628 1.7186 -309.155 -305.916 -304.721 -303.763
#>            97.5% n_eff  Rhat
#> alpha      3.172  2243 1.001
#> beta[1]    0.680  2503 1.000
#> beta[2]    0.452  1874 1.001
#> beta[3]   23.024  1871 1.002
#> beta[4]   -9.028  2273 1.000
#> sigma      4.996  2695 1.000
#> mu[1]     85.286  4000 1.000
#> mu[2]     88.241  3720 1.000
#> mu[3]     95.525  3273 1.000
#> mu[4]     82.752  4000 1.000
#> mu[5]     86.505  4000 1.000
#> mu[6]     61.814  2898 1.000
#> mu[7]     89.168  3444 1.000
#> mu[8]    101.770  2567 1.001
#> mu[9]     59.817  4000 1.001
#> mu[10]    91.461  4000 1.000
#> mu[11]    69.581  3154 1.000
#> mu[12]    98.508  2742 1.000
#> mu[13]    98.092  2813 1.000
#> mu[14]    72.633  3511 1.000
#> mu[15]    79.200  3527 1.000
#> mu[16]    47.731  2786 0.999
#> mu[17]    67.882  2041 1.001
#> mu[18]    74.960  2333 1.001
#> mu[19]    61.041  1952 1.001
#> mu[20]    93.967  3651 1.000
#> mu[21]    31.767  4000 1.000
#> mu[22]    36.475  4000 0.999
#> mu[23]    46.405  4000 1.000
#> mu[24]    23.826  4000 1.000
#> mu[25]    22.312  4000 0.999
#> mu[26]    44.236  2559 1.000
#> mu[27]    62.900  2560 1.000
#> mu[28]    34.632  3435 1.000
#> mu[29]    22.541  4000 0.999
#> mu[30]    37.252  3544 1.000
#> mu[31]    21.000  3627 0.999
#> mu[32]     9.154  2572 1.001
#> mu[33]    36.328  3638 1.000
#> mu[34]    14.047  3365 1.000
#> mu[35]    19.720  3587 1.000
#> mu[36]    21.266  4000 0.999
#> mu[37]    20.456  4000 0.999
#> mu[38]    21.177  3776 0.999
#> mu[39]    13.388  3257 1.000
#> mu[40]    17.900  3820 0.999
#> mu[41]    19.714  3383 0.999
#> mu[42]    20.695  4000 0.999
#> mu[43]    13.284  3292 1.000
#> mu[44]    39.315  2262 1.001
#> mu[45]    18.219  3096 0.999
#> lp__    -302.587  1625 1.004
#> 
#> $c_summary
#> , , chains = chain:1
#> 
#>          stats
#> parameter     mean     sd     2.5%      25%      50%      75%    97.5%
#>   alpha     -0.273 1.7465   -3.582   -1.478   -0.347    0.898    3.215
#>   beta[1]    0.599 0.0443    0.512    0.569    0.598    0.630    0.686
#>   beta[2]    0.347 0.0544    0.237    0.312    0.349    0.385    0.457
#>   beta[3]   16.478 3.2400   10.523   14.089   16.510   18.708   22.733
#>   beta[4]  -14.672 2.8448  -20.174  -16.561  -14.659  -12.731   -9.082
#>   sigma      4.570 0.2104    4.187    4.424    4.568    4.707    4.993
#>   mu[1]     83.189 1.0871   81.194   82.475   83.182   83.864   85.389
#>   mu[2]     85.703 1.2674   83.356   84.840   85.680   86.455   88.358
#>   mu[3]     93.056 1.2873   90.592   92.174   93.032   93.835   95.747
#>   mu[4]     80.388 1.2089   78.086   79.586   80.358   81.187   82.780
#>   mu[5]     84.387 1.0920   82.375   83.662   84.381   85.080   86.616
#>   mu[6]     57.949 2.0619   53.892   56.516   57.930   59.358   61.932
#>   mu[7]     86.817 1.2197   84.524   86.001   86.788   87.627   89.205
#>   mu[8]     98.827 1.5330   95.938   97.793   98.831   99.759  101.948
#>   mu[9]     55.380 2.2082   50.881   53.912   55.407   56.852   59.693
#>   mu[10]    89.176 1.1799   86.989   88.386   89.142   89.937   91.765
#>   mu[11]    67.048 1.2911   64.637   66.122   67.046   67.936   69.520
#>   mu[12]    95.738 1.4247   93.031   94.766   95.728   96.602   98.607
#>   mu[13]    95.391 1.4001   92.758   94.437   95.382   96.243   98.239
#>   mu[14]    69.922 1.3823   67.201   69.031   69.915   70.824   72.651
#>   mu[15]    76.544 1.3645   73.916   75.644   76.552   77.462   79.146
#>   mu[16]    42.362 2.7670   36.976   40.482   42.352   44.170   47.893
#>   mu[17]    63.563 2.1602   59.605   62.105   63.492   64.990   68.186
#>   mu[18]    71.574 1.7331   68.283   70.446   71.511   72.739   75.297
#>   mu[19]    56.630 2.1709   52.661   55.088   56.589   58.040   61.234
#>   mu[20]    91.379 1.3273   88.954   90.451   91.353   92.207   94.144
#>   mu[21]    27.421 2.2469   22.905   25.872   27.393   28.846   31.848
#>   mu[22]    32.892 1.8324   29.160   31.713   32.849   34.115   36.342
#>   mu[23]    42.639 1.8551   38.864   41.446   42.654   43.811   46.099
#>   mu[24]    19.781 2.0728   15.693   18.411   19.813   21.049   23.845
#>   mu[25]    20.287 1.0145   18.360   19.595   20.272   20.956   22.323
#>   mu[26]    41.409 1.4287   38.470   40.495   41.408   42.344   44.273
#>   mu[27]    57.944 2.6612   53.017   56.085   57.861   59.795   63.236
#>   mu[28]    32.393 1.1264   30.100   31.676   32.370   33.107   34.642
#>   mu[29]    20.538 1.0177   18.626   19.813   20.507   21.218   22.564
#>   mu[30]    34.751 1.3326   32.176   33.817   34.780   35.677   37.259
#>   mu[31]    18.779 1.1260   16.604   18.022   18.774   19.563   21.000
#>   mu[32]     6.349 1.4379    3.627    5.396    6.270    7.320    9.243
#>   mu[33]    33.901 1.2645   31.413   33.043   33.937   34.765   36.331
#>   mu[34]    11.714 1.1934    9.476   10.851   11.706   12.510   14.130
#>   mu[35]    17.509 1.1356   15.334   16.733   17.476   18.263   19.850
#>   mu[36]    19.245 1.0405   17.313   18.530   19.215   19.923   21.392
#>   mu[37]    18.335 1.0719   16.250   17.624   18.316   19.079   20.513
#>   mu[38]    19.030 1.0932   16.926   18.292   19.017   19.789   21.171
#>   mu[39]    11.019 1.2088    8.795   10.159   10.997   11.816   13.484
#>   mu[40]    15.749 1.0906   13.677   14.998   15.734   16.496   17.909
#>   mu[41]    17.330 1.2076   15.001   16.535   17.326   18.160   19.710
#>   mu[42]    18.587 1.0510   16.539   17.872   18.573   19.312   20.728
#>   mu[43]    10.864 1.2349    8.514    9.995   10.849   11.677   13.350
#>   mu[44]    36.404 1.4875   33.430   35.432   36.395   37.395   39.310
#>   mu[45]    15.630 1.3544   12.997   14.742   15.644   16.519   18.268
#>   lp__    -305.082 1.7492 -309.131 -305.956 -304.829 -303.811 -302.648
#> 
#> , , chains = chain:2
#> 
#>          stats
#> parameter     mean     sd     2.5%      25%      50%      75%    97.5%
#>   alpha     -0.336 1.7754   -4.005   -1.520   -0.331    0.848    3.029
#>   beta[1]    0.598 0.0407    0.521    0.569    0.596    0.628    0.679
#>   beta[2]    0.349 0.0532    0.239    0.313    0.349    0.385    0.453
#>   beta[3]   16.431 3.2666   10.200   14.212   16.509   18.488   23.105
#>   beta[4]  -14.646 2.8421  -20.106  -16.539  -14.643  -12.729   -9.187
#>   sigma      4.568 0.2119    4.167    4.426    4.555    4.721    5.001
#>   mu[1]     83.222 1.1105   80.940   82.509   83.249   83.924   85.314
#>   mu[2]     85.709 1.2323   83.171   84.915   85.731   86.576   88.103
#>   mu[3]     93.093 1.2978   90.524   92.258   93.126   93.960   95.558
#>   mu[4]     80.433 1.2278   78.023   79.672   80.449   81.239   82.890
#>   mu[5]     84.418 1.1132   82.129   83.695   84.435   85.128   86.493
#>   mu[6]     58.002 1.9738   54.102   56.708   57.968   59.335   61.772
#>   mu[7]     86.864 1.2474   84.355   86.034   86.869   87.706   89.254
#>   mu[8]     98.879 1.5517   95.893   97.841   98.866   99.947  101.862
#>   mu[9]     55.485 2.2927   51.219   53.972   55.352   57.035   60.272
#>   mu[10]    89.203 1.1811   86.737   88.418   89.251   89.999   91.433
#>   mu[11]    67.068 1.3009   64.589   66.182   67.038   67.891   69.581
#>   mu[12]    95.788 1.4472   92.931   94.842   95.794   96.779   98.599
#>   mu[13]    95.438 1.4206   92.636   94.513   95.446   96.387   98.203
#>   mu[14]    69.964 1.3707   67.293   69.066   69.977   70.835   72.668
#>   mu[15]    76.596 1.3629   73.923   75.706   76.584   77.479   79.348
#>   mu[16]    42.351 2.6233   37.081   40.532   42.387   44.170   47.541
#>   mu[17]    63.515 2.1323   59.299   62.001   63.580   64.878   67.740
#>   mu[18]    71.545 1.6972   68.304   70.331   71.626   72.700   74.859
#>   mu[19]    56.587 2.1740   52.353   55.078   56.659   57.996   60.887
#>   mu[20]    91.394 1.2889   88.647   90.524   91.408   92.261   93.853
#>   mu[21]    27.517 2.1919   23.001   26.097   27.549   28.966   31.887
#>   mu[22]    32.941 1.8152   29.364   31.786   32.868   34.089   36.437
#>   mu[23]    42.718 1.8905   39.095   41.449   42.680   43.966   46.602
#>   mu[24]    19.831 2.0136   15.870   18.494   19.848   21.117   23.747
#>   mu[25]    20.259 1.0624   18.326   19.490   20.274   20.980   22.383
#>   mu[26]    41.400 1.4851   38.463   40.397   41.394   42.350   44.424
#>   mu[27]    57.891 2.5800   52.873   56.130   57.787   59.665   63.223
#>   mu[28]    32.375 1.1900   30.093   31.600   32.363   33.112   34.810
#>   mu[29]    20.508 1.0713   18.574   19.714   20.517   21.247   22.640
#>   mu[30]    34.714 1.3790   32.106   33.770   34.714   35.643   37.362
#>   mu[31]    18.767 1.1189   16.685   17.984   18.757   19.476   21.101
#>   mu[32]     6.296 1.4725    3.323    5.285    6.322    7.277    9.090
#>   mu[33]    33.867 1.3181   31.352   32.987   33.865   34.768   36.485
#>   mu[34]    11.685 1.1963    9.493   10.838   11.677   12.466   14.113
#>   mu[35]    17.464 1.1953   15.134   16.596   17.474   18.310   19.869
#>   mu[36]    19.211 1.0948   17.230   18.419   19.223   19.959   21.370
#>   mu[37]    18.317 1.0842   16.334   17.591   18.291   19.006   20.648
#>   mu[38]    19.016 1.0968   16.972   18.265   18.988   19.707   21.291
#>   mu[39]    10.986 1.2202    8.732   10.134   10.975   11.786   13.486
#>   mu[40]    15.723 1.1085   13.750   14.948   15.694   16.476   18.036
#>   mu[41]    17.322 1.1767   15.152   16.508   17.324   18.076   19.815
#>   mu[42]    18.566 1.0732   16.574   17.826   18.539   19.261   20.830
#>   mu[43]    10.838 1.2217    8.638    9.955   10.830   11.638   13.369
#>   mu[44]    36.419 1.5142   33.443   35.431   36.468   37.397   39.417
#>   mu[45]    15.629 1.2889   13.207   14.728   15.614   16.438   18.223
#>   lp__    -305.064 1.7399 -309.226 -306.059 -304.800 -303.783 -302.564
#> 
#> , , chains = chain:3
#> 
#>          stats
#> parameter     mean     sd     2.5%      25%      50%      75%    97.5%
#>   alpha     -0.245 1.7515   -3.561   -1.516   -0.252    1.010    3.252
#>   beta[1]    0.597 0.0426    0.513    0.567    0.598    0.629    0.674
#>   beta[2]    0.346 0.0546    0.247    0.306    0.346    0.381    0.462
#>   beta[3]   16.693 3.3838   10.106   14.392   16.630   19.042   23.353
#>   beta[4]  -14.460 2.8932  -19.856  -16.338  -14.480  -12.604   -8.568
#>   sigma      4.579 0.2143    4.168    4.437    4.570    4.714    5.014
#>   mu[1]     83.192 1.1147   81.015   82.465   83.181   83.953   85.268
#>   mu[2]     85.705 1.2809   83.011   84.892   85.763   86.488   88.234
#>   mu[3]     93.027 1.2884   90.467   92.128   93.051   93.881   95.504
#>   mu[4]     80.396 1.2278   77.996   79.604   80.393   81.233   82.765
#>   mu[5]     84.386 1.1170   82.185   83.662   84.381   85.146   86.485
#>   mu[6]     58.024 2.0479   54.073   56.552   58.098   59.514   61.979
#>   mu[7]     86.806 1.2326   84.383   85.993   86.829   87.638   89.170
#>   mu[8]     98.777 1.5303   95.613   97.729   98.839   99.857  101.719
#>   mu[9]     55.370 2.3067   51.053   53.742   55.287   56.916   59.899
#>   mu[10]    89.162 1.1886   86.793   88.382   89.190   89.922   91.465
#>   mu[11]    67.104 1.3588   64.478   66.168   67.113   68.017   69.736
#>   mu[12]    95.698 1.4254   92.771   94.706   95.737   96.677   98.397
#>   mu[13]    95.352 1.4005   92.514   94.375   95.387   96.302   98.027
#>   mu[14]    69.964 1.4067   67.217   69.032   69.979   70.924   72.594
#>   mu[15]    76.563 1.3733   73.980   75.668   76.589   77.520   79.216
#>   mu[16]    42.420 2.8201   36.772   40.673   42.438   44.431   48.066
#>   mu[17]    63.645 2.2517   58.970   62.198   63.608   65.202   68.010
#>   mu[18]    71.627 1.7995   67.782   70.475   71.662   72.794   74.994
#>   mu[19]    56.732 2.2910   52.024   55.192   56.745   58.278   61.092
#>   mu[20]    91.361 1.3194   88.739   90.489   91.410   92.184   93.953
#>   mu[21]    27.499 2.2523   23.019   25.895   27.478   29.136   31.730
#>   mu[22]    32.964 1.9250   29.342   31.639   32.954   34.258   36.678
#>   mu[23]    42.675 1.9440   38.934   41.374   42.664   43.971   46.528
#>   mu[24]    19.893 2.1336   15.819   18.369   19.830   21.351   24.175
#>   mu[25]    20.243 0.9853   18.350   19.581   20.225   20.899   22.116
#>   mu[26]    41.296 1.4356   38.527   40.282   41.275   42.287   43.953
#>   mu[27]    57.791 2.5920   52.575   55.974   57.772   59.762   62.659
#>   mu[28]    32.309 1.1164   30.184   31.565   32.311   33.088   34.496
#>   mu[29]    20.495 0.9903   18.544   19.844   20.472   21.182   22.379
#>   mu[30]    34.665 1.3069   32.135   33.794   34.698   35.528   37.135
#>   mu[31]    18.736 1.0817   16.699   17.963   18.691   19.469   20.895
#>   mu[32]     6.354 1.4291    3.619    5.291    6.341    7.333    9.163
#>   mu[33]    33.817 1.2422   31.419   32.981   33.843   34.666   36.176
#>   mu[34]    11.697 1.1551    9.526   10.818   11.675   12.456   14.044
#>   mu[35]    17.478 1.1154   15.260   16.740   17.445   18.243   19.686
#>   mu[36]    19.206 1.0142   17.243   18.561   19.184   19.895   21.182
#>   mu[37]    18.296 1.0321   16.302   17.564   18.253   19.008   20.444
#>   mu[38]    18.987 1.0517   17.011   18.229   18.937   19.708   21.154
#>   mu[39]    11.005 1.1755    8.802   10.120   10.991   11.807   13.298
#>   mu[40]    15.719 1.0532   13.676   14.963   15.680   16.424   17.773
#>   mu[41]    17.290 1.1560   15.134   16.478   17.271   18.055   19.633
#>   mu[42]    18.547 1.0140   16.581   17.839   18.501   19.230   20.567
#>   mu[43]    10.848 1.1910    8.612    9.965   10.800   11.643   13.265
#>   mu[44]    36.301 1.4970   33.549   35.271   36.317   37.306   39.241
#>   mu[45]    15.594 1.2924   13.222   14.672   15.559   16.413   18.117
#>   lp__    -305.130 1.7550 -309.380 -306.032 -304.854 -303.840 -302.631
#> 
#> , , chains = chain:4
#> 
#>          stats
#> parameter     mean     sd     2.5%      25%       50%      75%    97.5%
#>   alpha     -0.144 1.6916   -3.612   -1.288   -0.0695    0.989    3.246
#>   beta[1]    0.597 0.0411    0.514    0.572    0.5959    0.625    0.679
#>   beta[2]    0.343 0.0512    0.250    0.307    0.3423    0.377    0.440
#>   beta[3]   16.841 3.0989   10.868   14.746   16.8713   18.958   22.758
#>   beta[4]  -14.490 2.7618  -19.896  -16.382  -14.3964  -12.591   -9.266
#>   sigma      4.557 0.1973    4.166    4.429    4.5541    4.679    4.962
#>   mu[1]     83.225 1.0058   81.142   82.585   83.2811   83.852   85.133
#>   mu[2]     85.769 1.1934   83.289   84.973   85.7763   86.561   88.147
#>   mu[3]     93.048 1.2215   90.662   92.246   93.1082   93.874   95.346
#>   mu[4]     80.416 1.1012   78.180   79.698   80.4268   81.150   82.597
#>   mu[5]     84.420 1.0149   82.295   83.773   84.4535   85.055   86.357
#>   mu[6]     58.049 1.8657   54.347   56.751   58.1326   59.300   61.617
#>   mu[7]     86.821 1.1331   84.532   86.050   86.8482   87.525   89.058
#>   mu[8]     98.778 1.4648   95.794   97.827   98.7768   99.765  101.537
#>   mu[9]     55.223 2.2063   50.847   53.813   55.2694   56.667   59.470
#>   mu[10]    89.199 1.1134   86.895   88.473   89.2594   89.927   91.342
#>   mu[11]    67.163 1.1823   64.948   66.366   67.1623   67.981   69.530
#>   mu[12]    95.703 1.3546   92.966   94.810   95.7020   96.634   98.242
#>   mu[13]    95.360 1.3312   92.625   94.472   95.3717   96.266   97.838
#>   mu[14]    69.996 1.2452   67.609   69.086   70.0372   70.825   72.341
#>   mu[15]    76.577 1.2319   74.147   75.756   76.5656   77.403   78.957
#>   mu[16]    42.423 2.6812   37.163   40.680   42.4185   44.348   47.491
#>   mu[17]    63.788 2.0474   59.709   62.445   63.8845   65.180   67.719
#>   mu[18]    71.742 1.6353   68.436   70.662   71.7618   72.871   74.843
#>   mu[19]    56.875 2.0653   52.736   55.465   56.9432   58.275   60.921
#>   mu[20]    91.411 1.2572   88.981   90.574   91.4599   92.229   93.847
#>   mu[21]    27.380 2.1404   23.015   26.056   27.3857   28.813   31.589
#>   mu[22]    32.900 1.8222   29.430   31.745   32.9799   34.096   36.377
#>   mu[23]    42.568 1.8560   38.926   41.339   42.6209   43.817   46.215
#>   mu[24]    19.836 2.0071   15.991   18.453   19.8217   21.219   23.659
#>   mu[25]    20.288 0.9844   18.393   19.594   20.3032   20.952   22.253
#>   mu[26]    41.305 1.3765   38.666   40.335   41.3098   42.238   44.058
#>   mu[27]    57.842 2.4851   52.803   56.064   57.8607   59.472   62.820
#>   mu[28]    32.334 1.0883   30.206   31.600   32.3099   33.075   34.583
#>   mu[29]    20.542 0.9869   18.642   19.866   20.5526   21.217   22.478
#>   mu[30]    34.712 1.2637   32.416   33.843   34.6562   35.613   37.234
#>   mu[31]    18.761 1.0835   16.620   18.049   18.7359   19.533   20.873
#>   mu[32]     6.438 1.3934    3.589    5.494    6.5225    7.389    9.101
#>   mu[33]    33.860 1.2037   31.643   33.053   33.7998   34.704   36.329
#>   mu[34]    11.748 1.1544    9.394   10.980   11.7778   12.566   13.889
#>   mu[35]    17.544 1.0965   15.332   16.805   17.5618   18.322   19.658
#>   mu[36]    19.259 1.0086   17.267   18.580   19.2692   19.966   21.164
#>   mu[37]    18.330 1.0362   16.319   17.649   18.3306   19.071   20.343
#>   mu[38]    19.016 1.0545   16.931   18.329   19.0145   19.775   21.082
#>   mu[39]    11.062 1.1709    8.620   10.280   11.0811   11.899   13.165
#>   mu[40]    15.764 1.0563   13.718   15.019   15.7748   16.516   17.795
#>   mu[41]    17.312 1.1562   15.046   16.530   17.2973   18.131   19.626
#>   mu[42]    18.584 1.0178   16.640   17.877   18.6085   19.292   20.577
#>   mu[43]    10.896 1.1914    8.490   10.075   10.9136   11.734   13.085
#>   mu[44]    36.284 1.4340   33.520   35.312   36.2728   37.187   39.210
#>   mu[45]    15.609 1.2874   13.173   14.706   15.5590   16.472   18.222
#>   lp__    -304.779 1.6064 -308.599 -305.579 -304.4984 -303.639 -302.573
```

### Convergence Diagnostics and Model Fit

-   **Convergence Diagnostics:** Is this the posterior distribution that you
    were looking for? These don't directly say anything about how "good" the
    model is in terms representing the data, they are only evaluating how well
    the sampler is doing at sampling the posterior distribution of the given
    model. If there are problems with these, then the sample results do not
    represent the posterior distribution, and your inferences will be biased.

    -   `mcse`:
    -   `n_eff`:
    -   `Rhat`
    -   `divergences`

-   **Model fit:** Is this statistical model appropriate for the data?
    Or better than other models?

    -   Posterior predictive checks

    -   Information criteria:

        -   WAIC
        -   Leave-one-out Cross-Validation
