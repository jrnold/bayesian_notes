
# Rare Events Logit


```r
library("rstan")
library("tidyverse")
```

There are two issues when estimating binary outcomes with an unbalanced outcome
variable (many-0's or many-1's), called rare events.

1.  Bias due to effective small sample size: The solution to this is the same as
    quasi-separation: a weakly informative prior on the coefficients.

1.  Case-control: Adding additional observations to the majority class adds little
    additional information. If it is costly to acquire training data, it is 
    better to acquire something closer to a balanced training set: approximately 
    equal numbers of 0's and 1's. The model can be adjusted for this bias.

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

In binary outcome variables, sometimes it is useful to sample on the dependent variable.
For example, @KingZeng2001a and @KingZeng2001b discuss applications with respect to conflicts in international relations.
For most country-pairs, for most years, there is no conflict.
If some data are costly to gather, it may be cost efficient to gather data for conflict-years and then randomly select a smaller number of non-conflict years on which to gather data.
The sample will no longer be representative, but the estimates can be corrected to account for the data-gerating process.

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
y_i& \sim \dbern(\pi_i) \\
\pi_i &= \invlogit(\eta_i) \\
\eta_i &= \tilde{\alpha} + X \beta  \\
\tilde{\alpha} &= \alpha - \ln \left(\frac{(1 - \tau) \bar{y}}{\tau (1 - \bar{y})}  \right) \\
\alpha &\sim \dnorm(0, 10)  \\
\beta &\sim \dnorm(0, 2.5)
\end{aligned}
$$
This is implemented in `relogit1.stan`:

```r
mod_relogit1 <- stan_model("stan/relogit1.stan")
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:200:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:1:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Core:531:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:2:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/LU:47:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:3:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Cholesky:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Jacobi:29:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:3:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Cholesky:43:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/QR:17:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Householder:27:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SVD:48:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Geometry:58:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Eigenvalues:58:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:36:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/operator_unary_plus.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/scal/fun/constants.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/constants/constants.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/convert_from_string.hpp:15:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/lexical_cast.hpp:32:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/lexical_cast/try_lexical_convert.hpp:42:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/lexical_cast/detail/converter_lexical.hpp:52:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/container/container_fwd.hpp:61:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/container/detail/std_fwd.hpp:27:1: warning: inline namespaces are a C++11 feature [-Wc++11-inline-namespace]
#> BOOST_MOVE_STD_NS_BEG
#> ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/detail/std_ns_begin.hpp:18:34: note: expanded from macro 'BOOST_MOVE_STD_NS_BEG'
#>    #define BOOST_MOVE_STD_NS_BEG _LIBCPP_BEGIN_NAMESPACE_STD
#>                                  ^
#> /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/__config:439:52: note: expanded from macro '_LIBCPP_BEGIN_NAMESPACE_STD'
#> #define _LIBCPP_BEGIN_NAMESPACE_STD namespace std {inline namespace _LIBCPP_NAMESPACE {
#>                                                    ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:26:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SparseCore:66:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:27:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/OrderingMethods:71:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SparseCholesky:43:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:32:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SparseQR:35:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:33:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/IterativeLinearSolvers:46:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d250cd9152.cpp:609:
#> In file included from /Users/jrnold/Library/R/3.4/library/rstan/include/rstan/rstaninc.hpp:3:
#> In file included from /Users/jrnold/Library/R/3.4/library/rstan/include/rstan/stan_fit.hpp:36:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/services/optimize/bfgs.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/optimization/bfgs.hpp:9:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/optimization/lbfgs_update.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/circular_buffer.hpp:54:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/circular_buffer/details.hpp:20:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/move.hpp:30:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/iterator.hpp:27:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/detail/iterator_traits.hpp:29:1: warning: inline namespaces are a C++11 feature [-Wc++11-inline-namespace]
#> BOOST_MOVE_STD_NS_BEG
#> ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/detail/std_ns_begin.hpp:18:34: note: expanded from macro 'BOOST_MOVE_STD_NS_BEG'
#>    #define BOOST_MOVE_STD_NS_BEG _LIBCPP_BEGIN_NAMESPACE_STD
#>                                  ^
#> /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/__config:439:52: note: expanded from macro '_LIBCPP_BEGIN_NAMESPACE_STD'
#> #define _LIBCPP_BEGIN_NAMESPACE_STD namespace std {inline namespace _LIBCPP_NAMESPACE {
#>                                                    ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:44:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:45:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file71d250cd9152.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:58:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> 19 warnings generated.
```

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

```r
mod_relogit2 <- stan_model("stan/relogit2.stan")
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/gevv_vvv_vari.hpp:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/var.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/config.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/config.hpp:39:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/config/compiler/clang.hpp:200:11: warning: 'BOOST_NO_CXX11_RVALUE_REFERENCES' macro redefined [-Wmacro-redefined]
#> #  define BOOST_NO_CXX11_RVALUE_REFERENCES
#>           ^
#> <command line>:6:9: note: previous definition is here
#> #define BOOST_NO_CXX11_RVALUE_REFERENCES 1
#>         ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:1:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Core:531:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:2:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/LU:47:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:3:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Cholesky:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Jacobi:29:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:3:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Cholesky:43:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/QR:17:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Householder:27:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:5:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SVD:48:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Geometry:58:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:14:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/matrix_vari.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat/fun/Eigen_NumTraits.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Dense:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Eigenvalues:58:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:36:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/operator_unary_plus.hpp:7:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/scal/fun/constants.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/constants/constants.hpp:13:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/math/tools/convert_from_string.hpp:15:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/lexical_cast.hpp:32:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/lexical_cast/try_lexical_convert.hpp:42:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/lexical_cast/detail/converter_lexical.hpp:52:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/container/container_fwd.hpp:61:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/container/detail/std_fwd.hpp:27:1: warning: inline namespaces are a C++11 feature [-Wc++11-inline-namespace]
#> BOOST_MOVE_STD_NS_BEG
#> ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/detail/std_ns_begin.hpp:18:34: note: expanded from macro 'BOOST_MOVE_STD_NS_BEG'
#>    #define BOOST_MOVE_STD_NS_BEG _LIBCPP_BEGIN_NAMESPACE_STD
#>                                  ^
#> /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/__config:439:52: note: expanded from macro '_LIBCPP_BEGIN_NAMESPACE_STD'
#> #define _LIBCPP_BEGIN_NAMESPACE_STD namespace std {inline namespace _LIBCPP_NAMESPACE {
#>                                                    ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:26:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SparseCore:66:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:27:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/OrderingMethods:71:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:29:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SparseCholesky:43:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:32:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/SparseQR:35:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:83:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/csr_extract_u.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/Sparse:33:
#> In file included from /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/IterativeLinearSolvers:46:
#> /Users/jrnold/Library/R/3.4/library/RcppEigen/include/Eigen/src/Core/util/ReenableStupidWarnings.h:10:30: warning: pragma diagnostic pop could not pop, no matching push [-Wunknown-pragmas]
#>     #pragma clang diagnostic pop
#>                              ^
#> In file included from file71d27a8f6682.cpp:595:
#> In file included from /Users/jrnold/Library/R/3.4/library/rstan/include/rstan/rstaninc.hpp:3:
#> In file included from /Users/jrnold/Library/R/3.4/library/rstan/include/rstan/stan_fit.hpp:36:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/services/optimize/bfgs.hpp:11:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/optimization/bfgs.hpp:9:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/optimization/lbfgs_update.hpp:6:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/circular_buffer.hpp:54:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/circular_buffer/details.hpp:20:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/move.hpp:30:
#> In file included from /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/iterator.hpp:27:
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/detail/iterator_traits.hpp:29:1: warning: inline namespaces are a C++11 feature [-Wc++11-inline-namespace]
#> BOOST_MOVE_STD_NS_BEG
#> ^
#> /Users/jrnold/Library/R/3.4/library/BH/include/boost/move/detail/std_ns_begin.hpp:18:34: note: expanded from macro 'BOOST_MOVE_STD_NS_BEG'
#>    #define BOOST_MOVE_STD_NS_BEG _LIBCPP_BEGIN_NAMESPACE_STD
#>                                  ^
#> /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/__config:439:52: note: expanded from macro '_LIBCPP_BEGIN_NAMESPACE_STD'
#> #define _LIBCPP_BEGIN_NAMESPACE_STD namespace std {inline namespace _LIBCPP_NAMESPACE {
#>                                                    ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:44:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:17: warning: unused function 'set_zero_all_adjoints' [-Wunused-function]
#>     static void set_zero_all_adjoints() {
#>                 ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core.hpp:45:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/core/set_zero_all_adjoints_nested.hpp:17:17: warning: 'static' function 'set_zero_all_adjoints_nested' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
#>     static void set_zero_all_adjoints_nested() {
#>                 ^
#> In file included from file71d27a8f6682.cpp:8:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/src/stan/model/model_header.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math.hpp:4:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/rev/mat.hpp:12:
#> In file included from /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat.hpp:58:
#> /Users/jrnold/Library/R/3.4/library/StanHeaders/include/stan/math/prim/mat/fun/autocorrelation.hpp:17:14: warning: function 'fft_next_good_size' is not needed and will not be emitted [-Wunneeded-internal-declaration]
#>       size_t fft_next_good_size(size_t N) {
#>              ^
#> 19 warnings generated.
```

## Questions

-   Compare the estimates and efficiency of the two methods.

-   How would you evalate the predictive distributions using cross-validation in case
    of unbalanced classes?

-   Suppose that there is uncertainty about the population proportion, $\tau$.
    Incorporate that uncertainty into the model by making $\tau$ a parameter
    and giving it a prior distribution.

-   In the prior correction method, instead of adjust the intercept prior to 
    sampling, it could be done after sampling. Calculate the corrected intercept
    to the generated quantities block. Does it change the estimates? Does it
    change the efficiency of sampling?

See the example for [`Zelig-relogit`](http://docs.zeligproject.org/en/latest/zelig-relogit.html)
