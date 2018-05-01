
# Model Comparison

Don't check, but compare.

-   Information criteria
-   Predictive accuracy

Model comparison based on predictive performance

## Continuous model expansion

Embed the model in a larger model

-   add new parameters
-   broaden the class of models, e.g. normal to a $t$
-   combine different models into a super-model that includes both as special cases
-   add new data. E.g. embed the data into a hierarchical model to draw strength from other data.

The old model is a $p(y, \theta)$ is embedded or replaced by $p(y, y^*, \theta, \phi)$:
$$
p(\theta, \phi | y, y^*) \propto p(\phi) p(\theta | \phi) p(y, y^* | \theta, \phi) 
$$
Requires specifying a set of priors, $p(\theta | \phi)$, and $p(\phi)$.

*Accounting for model choice*: Model choice is another avenue for overfitting the data, which is only partly alleviated by including all sources of uncertainty in Bayesian analysis.

*ALternative model formulations*: Often including more complex models is preferrable since it includes that uncertainty. However, computation limits this.

Use model checking and sensitivity analysis to choose to expand models.

## Posterior Predictive Criteria

Most of these notes summarize the more complete treatment in @GelmanHwangVehtari2013a and @VehtariGelmanGabry2015a.

### Summary and Advice 

Models can be compared using its *expected predictive accuracy* on new data. Ways to evaluate predictive accuracy:

-   log posterior predictive density: $\log p_post(\tilde{y})$. The log probability of observing new 
-   [scoring rules](https://en.wikipedia.org/wiki/Scoring_rule) or [loss functions](https://en.wikipedia.org/wiki/Loss_function) specific to the problem/research question

Several methods to estimate expected log posterior predictive density (elpd)

-   within-sample log-posterior density (biased, too optimistic)
-   information criteria: WAIC, DIC, AIC with correct the bias within-sample log-posterior density with a penalty (number of parameters)
-   cross-validation: estimate it using heldout data

What should you use? 

-   Use the Pareto Smoothed Importance Sampling LOO [@VehtariGelmanGabry2015a] implemented in the **[loo](https://cran.r-project.org/package=loo)** package:

    -   It is computationally efficient as it doesn't require completely 
        re-fitting the model, unlike actual cross-validation

    -   it is fully Bayesian, unlike AIC and DIC

    -   it seems to perform better than WAIC

    -   it provides indicators for when it is a poor approximation (unlike AIC, 
        DIC, and WAIC)

    -   next best approximation would be the WAIC. No reason to use AIC or DIC ever.

-   For observations which the PSIS-LOO has $\hat{k} > 0.7$ (the estimator has
    infinite variance) and there aren't too many, use LOO-CV.

-   If PSIS-LOO has many observations with with $k > 0.7$, then use LOO-CV or k-fold CV

-   If the likelihood doesn't easily partition into observations or LOO is not 
    an appropriate prediction task, use the appropriate CV method (block k-fold,
    partitioned k-fold, time-series k-fold, rolling forecasts, etc.)

### Expected Log Predictive Density

Let $f$ be the true model, $y$ be the observed data, and $\tilde{y}$ be future data or alternative data not used in fitting the model.
The out-of-sample predictive fit for new data is
$$
\log p_{post}(\tilde{y}_i) = -\log \E_{post}(p(\tilde{y}_i)) = \log \int p(\tilde{y}_i | \theta) p_{post}(\theta) d\,\theta
$$ 
where $p_{post}(\tilde{y}_i)$ is the predictive density for $\tilde{y}_i$ from $p_{post}(\theta)$. $\E_{post}$ is an expectation that averages over the values posterior distribution of $\theta$.

Since the future data $\tilde{y}_i$ are unknown, the **expected out-of-sample log predictive density** (elpd) is,
$$
\begin{aligned}[t]
\mathrm{elpd} &= \text{expected log predictive density for a new data point} \\
&= E_f(\log p_{post}(\tilde{y}_i)) \\
&= \int (\log p_{post}(\tilde{y}_i)) f(\tilde{y}_i) \,d\tilde{y}_i
\end{aligned}
$$
