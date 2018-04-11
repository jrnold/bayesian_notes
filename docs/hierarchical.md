
# Hierarchical Models

- *Hierarchical models:* often groups of parameters, $\{\theta_1, \dots, \theta_J\}$, are related.
- E.g. countries, states, counties, years, etc. Even the regression coefficients, $\beta_1, \dots, \beta_k$ seen the in the [Shrinkage and Regularization] chapter.
- We can treat those $\theta_j$ as drawn from a *population distribution*, $\theta_j \sim p(\theta)$.
- The prior distribution $p(\theta)$ is called a *hyperprior* and its parameters are *hyperparameters*


*Exchangeability:*

- parameters $(\theta_1, \dots, \theta_J)$ are *exchangeable* if $p(\theta_1, \dots, \theta_J)$ don't depend on the indexes.
- i.i.d. models are a special case of exchangeability.


## Example: Baseball Hits

@EfronMorris1975a analyzed data from 18 players in the 1970 season.
The goal was to predict the batting average of these 18 players from their first 45 at-bats for the remainder of the 1970 season.

The following example is based on @CarpenterGabryGoodrich2017a and the **[rstanarm](https://cran.r-project.org/package=rstanarm)** vignette [Hierarchical Partial Pooling for Repeated Binary Trials](https://cran.r-project.org/web/packages/rstanarm/vignettes/pooling.html).


The hitting data used in @EfronMorris1975a is included in **[rstanarm](https://cran.r-project.org/package=rstanarm)** as [rstanarm](https://www.rdocumentation.org/packages/rstanarm/topics/bball1970):

```r
data("bball1970", package = "rstanarm")
bball1970 <-
  mutate(bball1970,
         BatAvg1 = Hits / AB,
         BatAvg2 = RemainingHits / RemainingAB)
bball1970
#>        Player AB Hits RemainingAB RemainingHits BatAvg1 BatAvg2
#> 1    Clemente 45   18         367           127   0.400   0.346
#> 2    Robinson 45   17         426           127   0.378   0.298
#> 3      Howard 45   16         521           144   0.356   0.276
#> 4   Johnstone 45   15         275            61   0.333   0.222
#> 5       Berry 45   14         418           114   0.311   0.273
#> 6     Spencer 45   14         466           126   0.311   0.270
#> 7   Kessinger 45   13         586           155   0.289   0.265
#> 8    Alvarado 45   12         138            29   0.267   0.210
#> 9       Santo 45   11         510           137   0.244   0.269
#> 10    Swaboda 45   11         200            46   0.244   0.230
#> 11 Petrocelli 45   10         538           142   0.222   0.264
#> 12  Rodriguez 45   10         186            42   0.222   0.226
#> 13      Scott 45   10         435           132   0.222   0.303
#> 14      Unser 45   10         277            73   0.222   0.264
#> 15   Williams 45   10         591           195   0.222   0.330
#> 16 Campaneris 45    9         558           159   0.200   0.285
#> 17     Munson 45    8         408           129   0.178   0.316
#> 18      Alvis 45    7          70            14   0.156   0.200
```

Let $y_i$ be the number of hits in the first 45 at bats for player $i$,
$$
\begin{aligned}[t]
y_i & \sim \dbin(45, \mu_i),
\end{aligned}
$$
where $\mu_i \in (0, 1)$ is the player-specific batting average.
Priors will be placed on the log-odds parameter, $\eta \in \R$,
$$
\begin{aligned}[t]
\mu_i &\sim \frac{1}{1 + \exp(-\eta_i)} . \\
\end{aligned}
$$

This example considers three ways of modeling $\mu_i$:


1. **Complete Pooling:** All players have the same batting average parameter.
    $$
    \eta_i = \eta .
    $$
    The common (log-odds) batting average is given a weakly informative prior,
    $$
    \eta \sim \dnorm(0, 2.5)
    $$
    On the log odds scale, this places 95% of the probability mass between 0.7 and 99.3 on the proportion scale.
    
2. **Non-pooled:** Each players (log-odds) batting average is independent, with each assigned a separate weak prior.
    $$
    \begin{aligned}[t]
    \eta_i &\sim \dnorm(0, 2.5)
    \end{aligned}
    $$

3. **Partial-pooling:** Each player has a separate (log-odds) batting average, but these batting average parameters are drawn from a common normal distribution.
    $$
    \begin{aligned}[t]
    \eta_i &\sim \dnorm(0, \tau) \\
    \tau &\sim \dnorm(0, 1)
    \end{aligned}
    $$


```r
bball1970_data <- list(
  N = nrow(bball1970),
  k = bball1970$AB,
  y = bball1970$Hits,
  k_new = bball1970$RemainingAB,
  y_new = bball1970$RemainingHits
)
```
Create a list to store models:

```r
models <- list()
```


```r
models[["nopool"]] <- stan_model("stan/binomial-no-pooling.stan")
```

```r
models[["nopool"]]
```

<pre>
  <code class="stan">/* Binomial Model (No pooling)

  A binomial model for $i = 1, \dots, N$, no pooling:
  $$
  p(y_i | n_i, \mu_i) &\sim \mathsf{Binomial}(y_i | n_i, \mu_i) \\
  \mu_i &= \logit^{-1}(\eta_i) \\
  p(\eta_i) &\sim \mathsf{Normal}^+(0, 10)
  $$

*/
data {
  int N;
  int y[N];
  int k[N];
  // new data
  int y_new[N];
  int k_new[N];
}
parameters {
  vector[N] eta;
}
model {
  eta ~ normal(0., 10.);
  y ~ binomial_logit(k, eta);
}
generated quantities {
  int y_rep[N];
  vector[N] log_lik;
  vector[N] log_lik_new;
  vector<lower = 0., upper = 1.>[N] mu;
  mu = inv_logit(eta);
  for (n in 1:N) {
    y_rep[n] = binomial_rng(k[n], mu[n]);
    log_lik[n] = binomial_logit_lpmf(y[n] | k[n], eta[n]);
    log_lik_new[n] = binomial_logit_lpmf(y_new[n] | k_new[n], eta[n]);
  }
}</code>
</pre>



```r
models[["pool"]] <- stan_model("stan/binomial-complete-pooling.stan")
```

```r
models[["pool"]]
```

<pre>
  <code class="stan">/* Binomial Model

  A binomial model for $i = 1, \dots, N$, with complete pooling
  $$
  \begin{aligned}[t]
  p(y_i | n_i, \mu) &\sim \mathsf{Binomial}(n_i, \mu) \\
  \mu &= \logit^{-1}(\eta) \\
  p(\eta) &\sim \mathsf{Normal}^+(0, 10)
  \end{aligned}
  $$

*/
data {
  int N;
  int y[N];
  int k[N];
  // new data
  int y_new[N];
  int k_new[N];
}
parameters {
  real eta;
}
model {
  eta ~ normal(0., 10.);
  y ~ binomial_logit(k, eta);
}
generated quantities {
  int y_rep[N];
  vector[N] log_lik;
  vector[N] log_lik_new;
  real<lower = 0., upper = 1.> mu;
  mu = inv_logit(eta);
  for (n in 1:N) { //
    y_rep[n] = binomial_rng(k[n], mu);
    log_lik[n] = binomial_logit_lpmf(y[n] | k[n], eta);
    log_lik_new[n] = binomial_logit_lpmf(y_new[n] | k_new[n], eta);
  }
}</code>
</pre>


```r
models[["partial"]] <- stan_model("stan/binomial-partial-pooling-t.stan")
```

```r
models[["partial"]]
```

<pre>
  <code class="stan">/* Binomial Model

  A binomial model for $i = 1, \dots, N$, with partial pooling
  $$
  \begin{aligned}[t]
  p(y_i | n_i, \mu_i) &\sim \mathsf{Binomial}(y_i | n_i, \mu_i) \\
  \mu_i &= \logit^{-1}(\eta_i) \\
  p(\eta_i | \tau) &\sim \mathsf{Normal}(alpha, \tau) \\
  p(\tau) &\sim \mathsf{Normal}^+(0, 1) \\
  p(alpha) & \sim \mathsf{Normal}(0, 2.5) \\
  \end{aligned}
  $$

*/
data {
  int N;
  int y[N];
  int k[N];
  // new data
  int y_new[N];
  int k_new[N];
}
parameters {
  vector[N] eta;
  real alpha;
  real<lower = 0.> tau;
}
model {
  alpha ~ normal(0., 10.);
  tau ~ normal(0., 1);
  eta ~ normal(alpha, tau);
  y ~ binomial_logit(k, eta);
}
generated quantities {
  int y_rep[N];
  vector[N] log_lik;
  vector[N] log_lik_new;
  vector<lower = 0., upper = 1.>[N] mu;
  mu = inv_logit(eta);
  for (n in 1:N) { //
    y_rep[n] = binomial_rng(k[n], mu[n]);
    log_lik[n] = binomial_logit_lpmf(y[n] | k[n], eta[n]);
    log_lik_new[n] = binomial_logit_lpmf(y_new[n] | k_new[n], eta[n]);
  }
}</code>
</pre>

Sample from all three models a

```r
fits <- map(models, sampling, data = bball1970_data,
            refresh = -1) %>%
  set_names(names(models))
#> Warning: There were 2 divergent transitions after warmup. Increasing adapt_delta above 0.8 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#> Warning: There were 4 chains where the estimated Bayesian Fraction of Missing Information was low. See
#> http://mc-stan.org/misc/warnings.html#bfmi-low
#> Warning: Examine the pairs() plot to diagnose sampling problems
```

For each model calculate the posterior mean of $\mu$ for each player:

```r
bball1970 <-
  map2_df(names(fits), fits, 
     function(nm, fit) {
      mu <- broom::tidy(fit) %>% filter(str_detect(term, "^mu"))
      if (nrow(mu) == 1) {
        out <- tibble(estimate = rep(mu$estimate, 18L))
      } else {
        out <- select(mu, estimate)
      }
      out$model <- nm
      out$.id <- seq_len(nrow(out))
      out
     }) %>%
  spread(model, estimate) %>%
  bind_cols(bball1970)
```
The partially pooled estiamtes are shrunk towards the overall average, and are between the no-pooling and pooled estimates.

```r
select(bball1970,
       Player, nopool, partial, pool) %>% 
  mutate(Player = factor(Player, levels = Player)) %>%
  gather(variable, value, -Player) %>%
  ggplot(aes(y = value, x = factor(variable), group = Player)) +
  geom_point() +
  geom_line() +
  labs(x = "", y = expression(mu))
```

<img src="hierarchical_files/figure-html/unnamed-chunk-13-1.png" width="70%" style="display: block; margin: auto;" />
We can plot the actual batting averages (`BatAvg1` and `BatAvg2`) and the model estimates:

```r
select(bball1970,
       Player, nopool, partial, pool, BatAvg1, BatAvg2) %>%
  mutate(Player = factor(Player, levels = Player)) %>%
  gather(variable, value, -Player) %>%
  ggplot(aes(y = Player, x = value, colour = variable)) +
  geom_point() +
  labs(x = expression(mu), y = "")
```

<img src="hierarchical_files/figure-html/unnamed-chunk-14-1.png" width="70%" style="display: block; margin: auto;" />
The estimates of the no-pooling model is almost exactly the same as `BatAvg1`.
The out-of-sample batting averages `BatAvg2` show regression to the mean.

For these models, compare the overall out-of-sample performance by calculating the actual average out-of-sample log-pointwise predictive density (lppd), and the expected lppd using LOO-PSIS.
The LOO-PSIS estimates of the out-of-sample lppd are optimistic.
However, they still show the pooling and partial estimates as superior to the no-pooling estimates.
The actual out-of-sample average lppd for the partial pooled model is the best fitting.

```r
map2_df(names(fits), fits, 
     function(nm, fit) {
      loo <- loo(extract_log_lik(fit, "log_lik"))
      ll_new <- rstan::extract(fit)[["log_lik_new"]]
      tibble(model = nm,
             loo = loo$elpd_loo / bball1970_data$N,
             ll_out = mean(log(colMeans(exp(ll_new)))))
     })
#> # A tibble: 3 × 3
#>     model   loo ll_out
#>     <chr> <dbl>  <dbl>
#> 1  nopool -3.20  -4.60
#> 2    pool -2.58  -4.05
#> 3 partial -2.59  -3.99
```

To see why this is the case, plot the average errors for each observation in- and out-of-sample.
In-sample for the no-pooling model is zero, but it over-estimates (under-estimates) the players with the highest (lowest) batting averages in their first 45 at bats---this is regression to the mean.
In sample, the partially pooling model shrinks the estimates towards the mean and 
reducing error.
Out of sample, the errors of the partially pooled model are not much different than the no-pooling model, except that the extreme observations have lower errors.

```r
select(bball1970,
       Player, nopool, partial, pool, BatAvg1, BatAvg2) %>% 
  mutate(Player = as.integer(factor(Player, levels = Player))) %>%
  gather(variable, value, -Player, -matches("BatAvg")) %>%
  mutate(`In-sample Errors` = value - BatAvg1,
         `Out-of-sample Errors` = value - BatAvg2) %>%
  select(-matches("BatAvg"), -value) %>%
  gather(sample, error, -variable, -Player) %>%
  ggplot(aes(y = error, x = Player, colour = variable)) +
  geom_hline(yintercept = 0, colour = "white", size = 2) +
  geom_point() +
  geom_line() +
  facet_wrap(~ sample, ncol = 1) +
  theme(legend.position = "bottom")
```

<img src="hierarchical_files/figure-html/unnamed-chunk-16-1.png" width="70%" style="display: block; margin: auto;" />

Extensions:

- Redo this analysis with the [rstanarm](https://www.rdocumentation.org/packages/rstanarm/topics/bball2006) dataset with hits and at-bats for the entire 2006 AL season of MLB.
- Use a beta distribution for the prior of $\mu_i$. How would you specify the prior beta distribution so that it is uniformative?
- If you used the beta distribution, how would you specify the beta distribution as a function of the mean?
- The lowest batting average of the modern era is approximately 0.16 and the highest is approximately 0.4. Use this information for an informative prior distribuiton.
- There may be some truly exceptional players. Model this by replacing the normal prior for $\eta$ with a wide tailed distribution.
- The distribution of batting averages may be asymmetric - since there may be a few great players, but a player can only be so bad before they are relegated to the minor league. Find a skewed distribution to use as a prior.


References:

- Albert, Jim. [Revisiting Efron and Morris’s Baseball Study](https://baseballwithr.wordpress.com/2016/02/15/revisiting-efron-and-morriss-baseball-study/) Feb 15, 2016
- Bob Carpenter. [Hierarchical Bayesian Batting Ability, with Multiple Comparisons](https://lingpipe-blog.com/2009/11/04/hierarchicalbayesian-batting-ability-with-multiple-comparisons/). November 4, 2009.
- John Kruschke. [Shrinkage in multi-level hierarchical models](http://doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html). November 27, 2012.
- See @JensenMcShaneWyner2009a for an updated hierarchical model of baseball hitting

### Other Examples

- Rat Tumors - BDA
- Eight Schools - BDA

