
# Rare Events

## Prerequisites {-}


```r
library("rstan")
library("tidyverse")
```

## Introduction

There are two issues when estimating model with a binary outcomes and rare events.

1.  Bias due to an effective small sample size: The solution to this is the same as
    quasi-separation, a weakly informative prior on the coefficients, 
    as discussed in the [Separation] chapter.

1.  Case-control: Adding additional observations to the majority class adds little
    additional information. If it is costly to acquire training data, it is 
    better to acquire something closer to a balanced training set: approximately 
    equal numbers of 0's and 1's. The model can be adjusted for this bias.

## Finite-Sample Bias

The finite-sample size bias can handled by the use of weakly informative priors
as discussed in the chapter on separation.
The current best-practice in Stan is to use the following weakly informative priors
for the intercept and coefficients:
$$
\begin{aligned}
\alpha &\sim \dnorm(0, 10)\\
\beta_k &\sim \dnorm(0, 2.5)
\end{aligned}
$$
The Normal priors could be replaced by Student-$t$ priors with finite variance.

## Case Control

In binary outcome variables, sometimes it is useful to sample on the dependent variable.
For example, @KingZeng2001a and @KingZeng2001b discuss applications with respect to conflicts in international relations.
For most country-pairs, for most years, there is no conflict.
If some data are costly to gather, it may be cost efficient to gather data for conflict-years and then randomly select a smaller number of non-conflict years on which to gather data.
The sample will no longer be representative, but the estimates can be corrected to account for the data-generating process.

The reason this works well, is that if there are few 1's, additional 0's have little influence on the estimation (@KingZeng2001a).

@KingZeng2001a propose two corrections:

1.  Prior correction
1.  Weighting observations

The *prior correction* model adjust the intercept of the logit model to account for the difference between the sample and population proportions.
Note that
$$
\pi_i = \frac{1}{1 + \exp(-\alpha + \mat{X} \beta)}
$$
An unbalanced sample only affects the intercept.
If $\alpha$ is the intercept estimated on the sample, the prior corrected intercept $\tilde{\alpha}$ is,
$$
\tilde{\alpha} = \alpha - \ln \left(\frac{1 - \tau}{\tau} \frac{\bar{y}}{1 - \bar{y}} \right)
$$
This is a special case of using an *offset* in a generalized linear model.
Since this constant is added to all observations it will not affect the estimation of $\alpha$ and $\beta$, but it will adjust the predicted probabilities for observations in the sample and new observations.

Thus the complete specification of a prior-correction rare event logits with standard weakly informative priors is:
$$
\begin{aligned}[t]
y_i& \sim \dbernoulli(\pi_i) \\
\pi_i &= \invlogit(\eta_i) \\
\eta_i &= \tilde{\alpha} + X \beta  \\
\tilde{\alpha} &= \alpha - \ln \left(\frac{(1 - \tau) \bar{y}}{\tau (1 - \bar{y})}  \right) \\
\alpha &\sim \dnorm(0, 10)  \\
\beta &\sim \dnorm(0, 2.5)
\end{aligned}
$$
This is implemented in `relogit1.stan`:
<!--html_preserve--><pre class="stan">
<code>// relogit1.stan
// Rare-Events Logit Model with Prior Correction
data {
  // number of observations
  int N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];
  // need to pass mean of y since it is hard to cast integer to real types
  // in stan
  real y_mean;
  // number of columns in the design matrix X
  int K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  real<lower=0.> scale_beta;
  // rare-event logit correction
  // population proportion of outcomes
  real<lower=0., upper=1.> tau;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
transformed data {
  real correction;

  // log((y_mean) / (1 - ymean) * tau / (1 - tau))
  correction = log(y_mean) - log1m(y_mean) + log1m(tau) - log(tau);
}
parameters {
  // regression coefficient vector
  real alpha_raw;
  vector[K] beta;
}
transformed parameters {
  // sampling corrected intercept
  real alpha;
  // logit-scale means
  vector[N] eta;

  alpha = alpha_raw - correction;
  eta = alpha + X * beta;
}
model {
  // priors
  alpha_raw ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  // likelihood
  y ~ bernoulli_logit(eta);
}
generated quantities {
  // simulate data from the posterior
  // actually simulate proportions rather than the outcomes - these are easy
  // enough to create.
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;

  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = bernoulli_rng(inv_logit(eta[i]));
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = bernoulli_lpmf(y[i] | inv_logit(eta[i]));
  }
}</code>
</pre><!--/html_preserve-->

The *weighted likelihood* model weights the contributions of each observation to the likelihood with its probability of selection.
As before, let $\mean{y}$ be the sample proportion of 1s in outcome, and 
$\tau$ be the known population proportion.
The probability of selection, conditional on the outcome is,
$$
w_i = 
\begin{cases}
\tau / \bar{y} & \text{if } y_i = 1 \text{,} \\
(1 - \tau) / (1 - \bar{y}) & \text{if } y_i = 0 \text{.}
\end{cases} 
$$

The log-likelihood is written as a weighted log-likelihood:
$$
\begin{aligned}[t]
\log L_w(\beta | y) &= \sum_i w_i \log \dbern (\pi_i)
\end{aligned}
$$
In Stan, this can be implemented by directly weighting the log-posterior contributions of each observation.

This is implemented in `relogit2.stan`:
<!--html_preserve--><pre class="stan">
<code>// relogit2.stan
// Rare-Events Logit Model with Prior Correction
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
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  real<lower=0.> scale_beta;
  // rare-event logit correction
  // population proportion of outcomes
  real<lower=0., upper=1.> tau;
  // need to pass mean of y since it is hard to cast integer to real types
  // in stan
  real y_mean;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
transformed data {
  // weights
  vector[N] w;
  for (n in 1:N) {
    if (y[n]) {
      w[n] = tau / y_mean;
    } else {
      w[n] = (1. - tau) / (1. - y_mean);
    }
  }
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
}
transformed parameters {
  // logit-scale means
  vector[N] eta;

  eta = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0.0, scale_alpha);
  beta ~ normal(0.0, scale_beta);
  // manually calculate the weighted log-likelihood
  for (n in 1:N) {
    target += w[n] * bernoulli_logit_lpmf(y[n] | eta[n]);
  }
}
generated quantities {
  // simulate data from the posterior
  // actually simulate proportions rather than the outcomes - these are easy
  // enough to create.
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;

  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = bernoulli_rng(inv_logit(eta[i]));
  }
  // we don't need to weight the individual log-likelihoods
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = bernoulli_lpmf(y[i] | inv_logit(eta[i]));
  }
}</code>
</pre><!--/html_preserve-->

## Questions

-   Compare the estimates and efficiency of the two methods.

-   How would you evaluate the predictive distributions using cross-validation in case
    of unbalanced classes?

-   Suppose that there is uncertainty about the population proportion, $\tau$.
    Incorporate that uncertainty into the model by making $\tau$ a parameter
    and giving it a prior distribution.

-   In the prior correction method, instead of adjust the intercept prior to 
    sampling, it could be done after sampling. Calculate the corrected intercept
    to the generated quantities block. Does it change the estimates? Does it
    change the efficiency of sampling?

See the example for [`Zelig-relogit`](http://docs.zeligproject.org/en/latest/zelig-relogit.html)
