
---
output: html_document
editor_options: 
  chunk_output_type: console
---
# Distributions

The parameterizations and notations for distributions largely follow @BDA3 and @Stan2016a.

The Wikipedia [List of Probability Distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions) is a fairly complete reference.
Standard references of probability distributions are [@JohnsonKotzBalakrishnan1994a;@JohnsonKotzBalakrishnan1995a;@KotzBalakrishnanJohnson2000a;@JohnsonKotzBalakrishnan1997a;@WimmerAltmann1999a;@ForbesEtAl2010a;@Asquith2011a].

The [Probability Distributions](https://cran.r-project.org/web/views/Distributions.html) CRAN task view contains both links and descriptions of probability distributions and as such serves as a useful list of probability distributions.

"The Chart of Univariate Distribution Relationships" [@LeemisMcQueston2008a] is the classic chart of the relationships between univariate distributions.

There are a few variations of this chart online:

-   [Univariate Distribution Relationships](http://www.math.wm.edu/~leemis/chart/UDR/UDR.html)
-   [Diagram of distribution relationships](https://www.johndcook.com/blog/distribution_chart/)

<!--
### Beta Distribution

See [Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution).

$$
\dbeta(x | \alpha, \beta) = \frac{x^{\alpha - 1} (1 - x)^{\beta- 1}}{B(\alpha, \beta)}
$$
where $B(\alpha, \beta) = \Gamma(\alpha) \Gamma(\beta) / \Gamma(\alpha + \beta)$.

$$
\begin{aligned}[t]
\mu = \E(X) &= \frac{\alpha}{\alpha + \beta} \\
\sigma^2 = \Var(X) &= \frac{\alpha\beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}
\end{aligned}
$$

For modeling, $\alpha$ and $\beta$ are difficult to work with, so some alternative parameterizations may be useful [^param].

[^param]: <https://en.wikipedia.org/wiki/Beta_distribution>

Mean ($\mu$) and sample size ($\nu$),
$$
\begin{aligned}[t]
\alpha &= \mu \nu \\
\beta  &= (1 - \mu) \nu
\end{aligned}
$$
Mode ($\omega$) and concentration ($\kappa = \alpha + \beta$):
$$
\begin{aligned}[t]
\alpha &= \omega (\kappa - 2) + 1 \\
\beta  &= (1 - \omega) (\kappa - 2) + 1
\end{aligned}
$$
Mean ($\mu$) and variance ($\sigma^2$) is difficult because the variance is a function of the mean:
$$
\begin{aligned}[t]
\alpha &= \mu \left( \frac{\mu (1 - \mu)}{\sigma^2} - 1 \right), & \text{if } \sigma^2 < \mu (1 - \mu) , \\
\beta &= (1 - \mu) \left( \frac{\mu (1 - \mu)}{\sigma^2} - 1 \right), & \text{if } \sigma^2 < \mu (1 - \mu) .
\end{aligned}
$$


### Gamma Distribution

See [Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution).

Shape $k > 0$ and scale $\theta > 0$,
$$
\dgamma(x | k, \theta) = \frac{1}{\Gamma(k) \theta^k}x^{k - 1}\exp(- x / \theta) .
$$
with
$$
\begin{aligned}[t]
\mu = \E[X] &= k \theta \\
\sigma^2 = \Var[X] &= k \theta^2
\end{aligned}
$$
This can be reparameterized as,
$$
\begin{aligned}[t]
k &= \frac{\mu^2}{\sigma^2} \\
\theta &= \frac{\sigma^2}{\mu}
\end{aligned}
$$

Shape $\alpha > 0$ and rate $\beta > 0$,
$$
\dgamma(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha - 1}\exp(- \beta x) .
$$
with
$$
\begin{aligned}[t]
\mu = \E[X] &= \frac{\alpha}{\beta} \\
\sigma^2 = \Var[X] &= \frac{\alpha}{\beta^2}
\end{aligned}
$$
This can be reparameterized as,
$$
\begin{aligned}[t]
\alpha &= \frac{\mu^2}{\sigma^2} \\
\beta &= \frac{\mu}{\sigma^2}
\end{aligned}
$$
-->
