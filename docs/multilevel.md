
# Multilevel Models

Multilevel models are commonly used hierarchical model.
They extend (generalized) linear models to include coefficients that vary by discrete groups.

Suppose that there are $i = 1, dots, n$ observations, and each observation is in one of $j = 1, \dots, J$ groups.
Let $j[i]$ be the group for 
$$
\begin{aligned}[t]
y_i &\sim \dnorm(\alpha_{j[i]} + \beta_{j[i]} x_i, \sigma^2) \\
\begin{pmatrix}
\alpha_j \\
\beta_j
\end{pmatrix} 
& \sim
\dnorm
\left(
\begin{pmatrix}
\mu_\alpha \\
\mu_\beta
end{pmatrix},
\Omega
\right)
\end{aligned} .
$$

*Pooled model*: All coefficients are common between groups. This is equivalent to a linear model.
$$
\begin{aligned}[t]
y_i &\sim \dnorm(\alpha + \beta x_i, \sigma^2) \\
\begin{pmatrix}
\alpha \\
\betap
\end{pmatrix} 
&\sim
\dnorm
\left(
\begin{pmatrix}
\mu_\alpha \\
\mu_\beta
end{pmatrix},
\Omega
\right)
\end{aligned}
$$

*Pooled model*: All coefficients are common between groups. This is equivalent to a linear model.
$$
\begin{aligned}[t]
y_i &\sim \dnorm(\alpha + \beta x_i, \sigma^2) \\
\end{aligned}
$$
*Varying-intercept*: The slope coefficients ($\beta$) are common between groups, but the intercepts ($\alpha_j$) vary by group.
$$
\begin{aligned}[t]
y_i &\sim \dnorm(\alpha_{j[i]} + \beta x_i, \sigma^2) \\
\end{aligned}
$$
*Varying-slope model*: The groups share a common intercept, $\alpha$, but the slope coefficient ($\beta$), varies between groups. This is less common since it is hard to think of cases when it is appropriate.
$$
\begin{aligned}[t]
y_i &\sim \dnorm(\alpha + \beta_{j[i]} x_i, \sigma^2) \\
\end{aligned}
$$
These models go by different names in different literatures: *hierarchical (generalized) linear models*, *nested data models*, *mixed models*, *random coefficients*, *random-effects*, *random parameter models*,  *split-plot designs* [^mlm-names].

[^mlm-names]: <https://en.wikipedia.org/wiki/Multilevel_model>

The model can be extended to other cases:

- generalized linear models
- multiple parameters

One of the difficulties in these models is the prior to the covariance matrix, $\Omega$.


<div class="figure" style="text-align: center">
<img src="multilevel_files/figure-html/unnamed-chunk-2-1.png" alt="Visual representation of hierarchical models" width="70%" />
<p class="caption">(\#fig:unnamed-chunk-2)Visual representation of hierarchical models</p>
</div>


## Example: Radon

This example models the presence of radon in houess in Minnesota which appears in @GelmanHill2007a and @BDA3.
This is partly derived from a [Stan Case Study](http://mc-stan.org/documentation/case-studies/radon.html), which uses `PyStan` instead of **rstan**.


### Data

The [rstanarm](https://www.rdocumentation.org/packages/rstanarm/topics/radon) data is included in the **[rstanarm](https://cran.r-project.org/package=rstanarm)** package.

```r
data("radon", package = "rstanarm")
radon
#>     floor         county log_radon log_uranium
#> 1       1         AITKIN    0.8329     -0.6890
#> 2       0         AITKIN    0.8329     -0.6890
#> 3       0         AITKIN    1.0986     -0.6890
#> 4       0         AITKIN    0.0953     -0.6890
#> 5       0          ANOKA    1.1632     -0.8473
#> 6       0          ANOKA    0.9555     -0.8473
#> 7       0          ANOKA    0.4700     -0.8473
#> 8       0          ANOKA    0.0953     -0.8473
#> 9       0          ANOKA   -0.2231     -0.8473
#> 10      0          ANOKA    0.2624     -0.8473
#> 11      0          ANOKA    0.2624     -0.8473
#> 12      0          ANOKA    0.3365     -0.8473
#> 13      0          ANOKA    0.4055     -0.8473
#> 14      0          ANOKA   -0.6931     -0.8473
#> 15      0          ANOKA    0.1823     -0.8473
#> 16      0          ANOKA    1.5261     -0.8473
#> 17      0          ANOKA    0.3365     -0.8473
#> 18      0          ANOKA    0.7885     -0.8473
#> 19      0          ANOKA    1.7918     -0.8473
#> 20      0          ANOKA    1.2238     -0.8473
#> 21      0          ANOKA    0.6419     -0.8473
#> 22      0          ANOKA    1.7047     -0.8473
#> 23      0          ANOKA    1.8563     -0.8473
#> 24      0          ANOKA    0.6931     -0.8473
#> 25      0          ANOKA    1.9021     -0.8473
#> 26      0          ANOKA    1.1632     -0.8473
#> 27      0          ANOKA    1.9315     -0.8473
#> 28      0          ANOKA    1.9601     -0.8473
#> 29      0          ANOKA    2.0541     -0.8473
#> 30      0          ANOKA    1.6677     -0.8473
#> 31      0          ANOKA    1.5261     -0.8473
#> 32      0          ANOKA    1.5041     -0.8473
#> 33      0          ANOKA    1.0647     -0.8473
#> 34      0          ANOKA    2.1041     -0.8473
#> 35      0          ANOKA    0.5306     -0.8473
#> 36      0          ANOKA    1.4586     -0.8473
#> 37      0          ANOKA    1.7047     -0.8473
#> 38      0          ANOKA    1.4110     -0.8473
#> 39      0          ANOKA    0.8755     -0.8473
#> 40      0          ANOKA    1.0986     -0.8473
#> 41      0          ANOKA    0.4055     -0.8473
#> 42      0          ANOKA    1.2238     -0.8473
#> 43      0          ANOKA    1.0986     -0.8473
#> 44      1          ANOKA    0.6419     -0.8473
#> 45      1          ANOKA   -1.2040     -0.8473
#> 46      0          ANOKA    0.9163     -0.8473
#> 47      1          ANOKA    0.1823     -0.8473
#> 48      0          ANOKA    0.8329     -0.8473
#> 49      0          ANOKA   -0.3567     -0.8473
#> 50      0          ANOKA    0.5878     -0.8473
#> 51      0          ANOKA    1.0986     -0.8473
#> 52      0          ANOKA    0.8329     -0.8473
#> 53      0          ANOKA    0.5878     -0.8473
#> 54      0          ANOKA    0.4055     -0.8473
#> 55      0          ANOKA    0.6931     -0.8473
#> 56      0          ANOKA    0.6419     -0.8473
#> 57      1         BECKER    0.2624     -0.1135
#> 58      0         BECKER    1.4816     -0.1135
#> 59      1         BECKER    1.5261     -0.1135
#> 60      0       BELTRAMI    1.8563     -0.5934
#> 61      0       BELTRAMI    1.5476     -0.5934
#> 62      0       BELTRAMI    1.7579     -0.5934
#> 63      1       BELTRAMI    0.8329     -0.5934
#> 64      1       BELTRAMI   -0.6931     -0.5934
#> 65      1       BELTRAMI    1.5476     -0.5934
#> 66      1       BELTRAMI    1.5041     -0.5934
#> 67      0         BENTON    1.9021     -0.1429
#> 68      0         BENTON    1.0296     -0.1429
#> 69      1         BENTON    1.0986     -0.1429
#> 70      0         BENTON    1.0986     -0.1429
#> 71      0       BIGSTONE    1.9879      0.3871
#> 72      0       BIGSTONE    1.6292      0.3871
#> 73      0       BIGSTONE    0.9933      0.3871
#> 74      0      BLUEEARTH    1.6292      0.2716
#> 75      0      BLUEEARTH    2.5726      0.2716
#> 76      0      BLUEEARTH    1.9879      0.2716
#> 77      0      BLUEEARTH    1.9315      0.2716
#> 78      0      BLUEEARTH    2.5572      0.2716
#> 79      1      BLUEEARTH    1.7750      0.2716
#> 80      0      BLUEEARTH    2.2618      0.2716
#> 81      0      BLUEEARTH    1.8083      0.2716
#> 82      0      BLUEEARTH    1.3610      0.2716
#> 83      1      BLUEEARTH    2.6672      0.2716
#> 84      0      BLUEEARTH    0.6419      0.2716
#> 85      0      BLUEEARTH    1.9459      0.2716
#> 86      0      BLUEEARTH    1.5686      0.2716
#> 87      0      BLUEEARTH    2.2618      0.2716
#> 88      1          BROWN    0.9555      0.2776
#> 89      0          BROWN    1.9169      0.2776
#> 90      1          BROWN    1.4110      0.2776
#> 91      0          BROWN    2.3224      0.2776
#> 92      0        CARLTON    0.8329     -0.3323
#> 93      0        CARLTON    0.6419     -0.3323
#> 94      0        CARLTON    1.2528     -0.3323
#> 95      1        CARLTON    1.7405     -0.3323
#> 96      0        CARLTON    1.4816     -0.3323
#> 97      0        CARLTON    1.3863     -0.3323
#> 98      0        CARLTON    0.3365     -0.3323
#> 99      0        CARLTON    1.4586     -0.3323
#> 100     0        CARLTON   -0.1054     -0.3323
#> 101     0        CARLTON    0.7419     -0.3323
#> 102     0         CARVER    0.5306      0.0959
#> 104     0         CARVER    2.5649      0.0959
#> 106     1         CARVER    2.6946      0.0959
#> 108     1         CARVER    1.5686      0.0959
#> 110     0         CARVER    2.2721      0.0959
#> 112     1         CARVER   -2.3026      0.0959
#> 114     0           CASS    1.3350     -0.6082
#> 115     0           CASS    2.0149     -0.6082
#> 116     0           CASS    0.6931     -0.6082
#> 117     0           CASS    1.6864     -0.6082
#> 118     0           CASS    1.4110     -0.6082
#> 119     0       CHIPPEWA    2.0541      0.2737
#> 120     0       CHIPPEWA    0.4055      0.2737
#> 121     0       CHIPPEWA    2.3125      0.2737
#> 122     0       CHIPPEWA    2.2513      0.2737
#> 123     0        CHISAGO   -0.1054     -0.7353
#> 124     0        CHISAGO    1.5041     -0.7353
#> 125     0        CHISAGO    1.6292     -0.7353
#> 126     0        CHISAGO    0.7885     -0.7353
#> 127     0        CHISAGO    0.5878     -0.7353
#> 128     0        CHISAGO    2.1041     -0.7353
#> 129     1           CLAY    0.0000      0.3438
#> 130     0           CLAY    2.5649      0.3438
#> 131     0           CLAY    0.9933      0.3438
#> 132     1           CLAY    1.2809      0.3438
#> 133     0           CLAY    3.2847      0.3438
#> 134     0           CLAY    0.4700      0.3438
#> 135     0           CLAY    2.5726      0.3438
#> 136     0           CLAY    2.1861      0.3438
#> 137     0           CLAY    2.9755      0.3438
#> 138     1           CLAY    0.9555      0.3438
#> 139     0           CLAY    2.2083      0.3438
#> 140     0           CLAY    2.5802      0.3438
#> 141     0           CLAY    1.3083      0.3438
#> 142     1           CLAY    1.9459      0.3438
#> 143     1     CLEARWATER    1.5892     -0.0599
#> 144     0     CLEARWATER    1.2528     -0.0599
#> 145     1     CLEARWATER    0.0000     -0.0599
#> 146     0     CLEARWATER    1.2528     -0.0599
#> 147     0           COOK    1.0296     -0.5050
#> 148     0           COOK    0.4055     -0.5050
#> 149     0     COTTONWOOD    1.9315      0.3396
#> 150     1     COTTONWOOD    2.4159      0.3396
#> 151     1     COTTONWOOD   -2.3026      0.3396
#> 152     1     COTTONWOOD    0.9555      0.3396
#> 153     0       CROWWING    0.6419     -0.6334
#> 154     0       CROWWING    0.5306     -0.6334
#> 155     1       CROWWING    0.0953     -0.6334
#> 156     0       CROWWING    0.0000     -0.6334
#> 157     0       CROWWING    1.0986     -0.6334
#> 158     0       CROWWING    1.5041     -0.6334
#> 159     0       CROWWING    0.4700     -0.6334
#> 160     0       CROWWING    1.4351     -0.6334
#> 161     1       CROWWING    0.9555     -0.6334
#> 162     1       CROWWING    1.9169     -0.6334
#> 163     0       CROWWING    1.4816     -0.6334
#> 164     0       CROWWING    1.7228     -0.6334
#> 165     0         DAKOTA    1.3083     -0.0241
#> 166     0         DAKOTA    1.0647     -0.0241
#> 167     0         DAKOTA    2.6878     -0.0241
#> 168     0         DAKOTA    1.9169     -0.0241
#> 169     0         DAKOTA    2.0919     -0.0241
#> 170     0         DAKOTA    0.9933     -0.0241
#> 171     1         DAKOTA    1.0647     -0.0241
#> 172     0         DAKOTA    1.5041     -0.0241
#> 173     1         DAKOTA    0.5878     -0.0241
#> 174     0         DAKOTA    0.7419     -0.0241
#> 175     0         DAKOTA    0.7419     -0.0241
#> 176     0         DAKOTA    0.4700     -0.0241
#> 177     0         DAKOTA    2.2721     -0.0241
#> 178     0         DAKOTA    2.1041     -0.0241
#> 179     0         DAKOTA    1.2809     -0.0241
#> 180     1         DAKOTA   -0.1054     -0.0241
#> 181     0         DAKOTA    1.6487     -0.0241
#> 182     0         DAKOTA    1.1939     -0.0241
#> 183     0         DAKOTA    2.3888     -0.0241
#> 184     0         DAKOTA    2.1163     -0.0241
#> 185     0         DAKOTA    1.8563     -0.0241
#> 186     0         DAKOTA    1.5892     -0.0241
#> 187     0         DAKOTA    1.8083     -0.0241
#> 188     0         DAKOTA    0.1823     -0.0241
#> 189     0         DAKOTA    2.1748     -0.0241
#> 190     0         DAKOTA    2.1861     -0.0241
#> 191     0         DAKOTA    1.9315     -0.0241
#> 192     0         DAKOTA    0.8755     -0.0241
#> 193     0         DAKOTA    0.5306     -0.0241
#> 194     0         DAKOTA    1.0647     -0.0241
#> 195     0         DAKOTA    1.8871     -0.0241
#> 196     0         DAKOTA    0.5878     -0.0241
#> 197     0         DAKOTA    1.5476     -0.0241
#> 198     0         DAKOTA    1.2238     -0.0241
#> 199     0         DAKOTA    1.5041     -0.0241
#> 200     0         DAKOTA    3.0587     -0.0241
#> 201     0         DAKOTA    2.2192     -0.0241
#> 202     0         DAKOTA    0.0000     -0.0241
#> 203     0         DAKOTA    1.6094     -0.0241
#> 204     0         DAKOTA    1.6292     -0.0241
#> 205     0         DAKOTA    0.1823     -0.0241
#> 206     0         DAKOTA    2.0412     -0.0241
#> 207     0         DAKOTA    1.7047     -0.0241
#> 208     0         DAKOTA    1.3083     -0.0241
#> 209     0         DAKOTA    1.6094     -0.0241
#> 210     0         DAKOTA    1.5686     -0.0241
#> 211     0         DAKOTA    0.4055     -0.0241
#> 212     0         DAKOTA    1.2528     -0.0241
#> 213     0         DAKOTA    1.4586     -0.0241
#> 214     0         DAKOTA    0.9555     -0.0241
#> 215     0         DAKOTA    0.4055     -0.0241
#> 216     0         DAKOTA    0.4055     -0.0241
#> 217     0         DAKOTA    0.6931     -0.0241
#> 218     0         DAKOTA    1.5892     -0.0241
#> 219     1         DAKOTA    0.4055     -0.0241
#> 220     0         DAKOTA    1.3610     -0.0241
#> 221     0         DAKOTA    2.1861     -0.0241
#> 222     0         DAKOTA    1.4816     -0.0241
#> 223     0         DAKOTA    1.5041     -0.0241
#> 224     0         DAKOTA    1.5261     -0.0241
#> 225     0         DAKOTA    0.8329     -0.0241
#> 226     0         DAKOTA   -0.5108     -0.0241
#> 227     0         DAKOTA    1.7750     -0.0241
#> 228     0          DODGE    1.7047      0.2639
#> 229     0          DODGE    1.9879      0.2639
#> 230     0          DODGE    1.7579      0.2639
#> 231     0        DOUGLAS    2.0149      0.1557
#> 232     0        DOUGLAS    1.5892      0.1557
#> 233     0        DOUGLAS    1.9315      0.1557
#> 234     0        DOUGLAS    1.8718      0.1557
#> 235     1        DOUGLAS    1.3350      0.1557
#> 236     0        DOUGLAS    1.7228      0.1557
#> 237     0        DOUGLAS    2.0669      0.1557
#> 238     0        DOUGLAS    1.5041      0.1557
#> 239     0        DOUGLAS    1.0296      0.1557
#> 240     0      FARIBAULT    1.2528      0.2950
#> 241     0      FARIBAULT    1.4586      0.2950
#> 242     0      FARIBAULT    0.8755      0.2950
#> 243     1      FARIBAULT    0.3365      0.2950
#> 244     0      FARIBAULT    1.6677      0.2950
#> 245     0      FARIBAULT   -1.6094      0.2950
#> 246     1       FILLMORE    0.9555      0.4149
#> 247     0       FILLMORE    1.1939      0.4149
#> 248     0       FREEBORN    1.1939      0.2242
#> 249     0       FREEBORN    2.2721      0.2242
#> 250     0       FREEBORN    1.4586      0.2242
#> 251     0       FREEBORN    2.2083      0.2242
#> 252     0       FREEBORN    1.8563      0.2242
#> 253     0       FREEBORN    3.4874      0.2242
#> 254     0       FREEBORN    2.5878      0.2242
#> 255     1       FREEBORN    0.8329      0.2242
#> 256     1       FREEBORN    1.7405      0.2242
#> 257     0        GOODHUE    2.6672      0.1966
#> 258     1        GOODHUE    1.9459      0.1966
#> 259     0        GOODHUE    2.0412      0.1966
#> 260     1        GOODHUE    2.2925      0.1966
#> 261     0        GOODHUE    0.9933      0.1966
#> 262     0        GOODHUE    3.7751      0.1966
#> 263     0        GOODHUE    1.6094      0.1966
#> 264     0        GOODHUE    1.6094      0.1966
#> 265     0        GOODHUE    1.2809      0.1966
#> 266     0        GOODHUE    1.5892      0.1966
#> 267     0        GOODHUE    1.7405      0.1966
#> 268     0        GOODHUE    1.2809      0.1966
#> 269     0        GOODHUE    1.3863      0.1966
#> 270     0        GOODHUE    1.9169      0.1966
#> 271     0       HENNEPIN    2.0794     -0.0965
#> 272     0       HENNEPIN    1.2238     -0.0965
#> 273     1       HENNEPIN    0.7885     -0.0965
#> 274     0       HENNEPIN    0.5306     -0.0965
#> 275     0       HENNEPIN    1.4110     -0.0965
#> 276     0       HENNEPIN    0.6419     -0.0965
#> 277     0       HENNEPIN    0.9555     -0.0965
#> 278     0       HENNEPIN    2.4248     -0.0965
#> 279     0       HENNEPIN    0.9933     -0.0965
#> 280     0       HENNEPIN    1.3863     -0.0965
#> 281     0       HENNEPIN    2.0149     -0.0965
#> 282     0       HENNEPIN    0.3365     -0.0965
#> 283     0       HENNEPIN    0.0000     -0.0965
#> 284     0       HENNEPIN   -0.6931     -0.0965
#> 285     1       HENNEPIN    0.9555     -0.0965
#> 286     0       HENNEPIN    1.8083     -0.0965
#> 287     0       HENNEPIN    0.7419     -0.0965
#> 288     0       HENNEPIN    1.7047     -0.0965
#> 289     0       HENNEPIN    1.1314     -0.0965
#> 290     0       HENNEPIN    1.0986     -0.0965
#> 291     0       HENNEPIN    1.7228     -0.0965
#> 292     0       HENNEPIN    1.4351     -0.0965
#> 293     0       HENNEPIN    1.3863     -0.0965
#> 294     0       HENNEPIN    2.7081     -0.0965
#> 295     0       HENNEPIN    1.9879     -0.0965
#> 296     0       HENNEPIN    0.8755     -0.0965
#> 297     1       HENNEPIN    1.0647     -0.0965
#> 298     0       HENNEPIN    1.5041     -0.0965
#> 299     0       HENNEPIN    0.4700     -0.0965
#> 300     0       HENNEPIN    2.1633     -0.0965
#> 301     0       HENNEPIN    1.7405     -0.0965
#> 302     0       HENNEPIN    2.1633     -0.0965
#> 303     0       HENNEPIN    1.3610     -0.0965
#> 304     0       HENNEPIN    0.6419     -0.0965
#> 305     0       HENNEPIN    0.6931     -0.0965
#> 306     0       HENNEPIN    1.7228     -0.0965
#> 307     0       HENNEPIN    0.9555     -0.0965
#> 308     1       HENNEPIN   -0.1054     -0.0965
#> 309     0       HENNEPIN    0.7885     -0.0965
#> 310     0       HENNEPIN    1.0647     -0.0965
#> 311     0       HENNEPIN    1.3863     -0.0965
#> 312     0       HENNEPIN    1.4816     -0.0965
#> 313     0       HENNEPIN    1.5686     -0.0965
#> 314     0       HENNEPIN    1.0647     -0.0965
#> 315     0       HENNEPIN    1.4351     -0.0965
#> 316     0       HENNEPIN    0.5306     -0.0965
#> 317     0       HENNEPIN    1.4816     -0.0965
#> 318     1       HENNEPIN   -0.2231     -0.0965
#> 319     0       HENNEPIN    1.7228     -0.0965
#> 320     1       HENNEPIN    1.2238     -0.0965
#> 321     0       HENNEPIN    1.7228     -0.0965
#> 322     0       HENNEPIN    0.9555     -0.0965
#> 323     0       HENNEPIN    1.0296     -0.0965
#> 324     0       HENNEPIN    2.1401     -0.0965
#> 325     0       HENNEPIN    1.2238     -0.0965
#> 326     0       HENNEPIN    1.1939     -0.0965
#> 327     0       HENNEPIN    2.1633     -0.0965
#> 328     0       HENNEPIN    0.5878     -0.0965
#> 329     0       HENNEPIN    1.7579     -0.0965
#> 330     0       HENNEPIN    2.5726     -0.0965
#> 331     0       HENNEPIN    1.0296     -0.0965
#> 332     0       HENNEPIN    1.5686     -0.0965
#> 333     0       HENNEPIN    1.7405     -0.0965
#> 334     0       HENNEPIN    2.6319     -0.0965
#> 335     1       HENNEPIN    2.0412     -0.0965
#> 336     0       HENNEPIN    1.7579     -0.0965
#> 337     0       HENNEPIN    1.5476     -0.0965
#> 338     0       HENNEPIN    2.0412     -0.0965
#> 339     0       HENNEPIN    0.9933     -0.0965
#> 340     0       HENNEPIN    1.5261     -0.0965
#> 341     0       HENNEPIN    1.7918     -0.0965
#> 342     0       HENNEPIN    0.8329     -0.0965
#> 343     0       HENNEPIN    0.9163     -0.0965
#> 344     0       HENNEPIN    1.4110     -0.0965
#> 345     0       HENNEPIN    1.5476     -0.0965
#> 346     0       HENNEPIN    1.5476     -0.0965
#> 347     0       HENNEPIN    2.3979     -0.0965
#> 348     0       HENNEPIN    2.0412     -0.0965
#> 349     0       HENNEPIN    1.1314     -0.0965
#> 350     0       HENNEPIN    0.4700     -0.0965
#> 351     1       HENNEPIN    0.5306     -0.0965
#> 352     0       HENNEPIN    2.8094     -0.0965
#> 353     0       HENNEPIN    1.1632     -0.0965
#> 354     0       HENNEPIN    1.6487     -0.0965
#> 355     0       HENNEPIN    1.6094     -0.0965
#> 356     0       HENNEPIN    1.8083     -0.0965
#> 357     1       HENNEPIN    0.0000     -0.0965
#> 358     0       HENNEPIN    0.6419     -0.0965
#> 359     0       HENNEPIN    1.3863     -0.0965
#> 360     0       HENNEPIN    1.7405     -0.0965
#> 361     1       HENNEPIN   -0.6931     -0.0965
#> 362     0       HENNEPIN    0.9933     -0.0965
#> 363     0       HENNEPIN    1.3083     -0.0965
#> 364     0       HENNEPIN    1.8405     -0.0965
#> 365     0       HENNEPIN    3.1655     -0.0965
#> 366     0       HENNEPIN    1.3863     -0.0965
#> 367     0       HENNEPIN    1.0986     -0.0965
#> 368     0       HENNEPIN    1.1314     -0.0965
#> 369     0       HENNEPIN    1.5686     -0.0965
#> 370     0       HENNEPIN    1.1314     -0.0965
#> 371     0       HENNEPIN    1.4586     -0.0965
#> 372     0       HENNEPIN    1.3610     -0.0965
#> 373     0       HENNEPIN    1.1314     -0.0965
#> 374     0       HENNEPIN    1.4816     -0.0965
#> 375     1       HENNEPIN    1.0986     -0.0965
#> 376     0        HOUSTON    1.2528      0.5035
#> 377     0        HOUSTON    2.1518      0.5035
#> 378     0        HOUSTON    2.2083      0.5035
#> 379     1        HOUSTON    1.5892      0.5035
#> 380     0        HOUSTON    1.3083      0.5035
#> 381     1        HOUSTON    0.8329      0.5035
#> 382     1        HUBBARD    1.0647     -0.4006
#> 383     1        HUBBARD   -0.1054     -0.4006
#> 384     0        HUBBARD    0.4700     -0.4006
#> 385     0        HUBBARD    1.5476     -0.4006
#> 386     1        HUBBARD    1.3350     -0.4006
#> 387     0         ISANTI    1.3083     -0.7519
#> 388     0         ISANTI    1.1314     -0.7519
#> 389     0         ISANTI    0.8329     -0.7519
#> 390     0         ITASCA    0.6931     -0.6633
#> 391     0         ITASCA    0.9933     -0.6633
#> 392     0         ITASCA    0.6419     -0.6633
#> 393     0         ITASCA    0.9163     -0.6633
#> 394     0         ITASCA    1.4816     -0.6633
#> 395     0         ITASCA    0.9933     -0.6633
#> 396     0         ITASCA    0.1823     -0.6633
#> 397     0         ITASCA    1.2238     -0.6633
#> 398     0         ITASCA    0.9555     -0.6633
#> 399     0         ITASCA    2.2513     -0.6633
#> 400     0         ITASCA    0.3365     -0.6633
#> 401     0        JACKSON    2.1401      0.3090
#> 402     0        JACKSON    1.6292      0.3090
#> 403     0        JACKSON    1.0986      0.3090
#> 404     0        JACKSON    2.5802      0.3090
#> 405     0        JACKSON    2.7344      0.3090
#> 406     0        KANABEC    0.6419     -0.0534
#> 407     0        KANABEC    1.3610     -0.0534
#> 408     0        KANABEC    2.0794     -0.0534
#> 409     0        KANABEC    0.9933     -0.0534
#> 410     0      KANDIYOHI    2.4336      0.1097
#> 411     0      KANDIYOHI    1.4351      0.1097
#> 412     0      KANDIYOHI    2.5177      0.1097
#> 413     0      KANDIYOHI    1.9169      0.1097
#> 414     0        KITTSON    1.9459     -0.0078
#> 415     1        KITTSON    1.5261     -0.0078
#> 416     1        KITTSON    0.0000     -0.0078
#> 417     0    KOOCHICHING    0.5878     -0.8818
#> 418     0    KOOCHICHING    0.4055     -0.8818
#> 419     1    KOOCHICHING    0.7419     -0.8818
#> 420     1    KOOCHICHING    0.0953     -0.8818
#> 421     0    KOOCHICHING    0.0953     -0.8818
#> 422     1    KOOCHICHING    1.0647     -0.8818
#> 423     1    KOOCHICHING    0.3365     -0.8818
#> 424     1    LACQUIPARLE    2.4336      0.3110
#> 426     0    LACQUIPARLE    2.7788      0.3110
#> 428     1           LAKE    0.3365     -0.6916
#> 429     0           LAKE    0.3365     -0.6916
#> 430     0           LAKE    0.5306     -0.6916
#> 431     0           LAKE    0.0000     -0.6916
#> 432     0           LAKE    1.0647     -0.6916
#> 433     0           LAKE   -0.5108     -0.6916
#> 434     0           LAKE    0.4700     -0.6916
#> 435     0           LAKE    1.9741     -0.6916
#> 436     0           LAKE   -0.5108     -0.6916
#> 437     0 LAKEOFTHEWOODS    2.3224     -0.6817
#> 438     1 LAKEOFTHEWOODS    1.4816     -0.6817
#> 439     0 LAKEOFTHEWOODS    1.2238     -0.6817
#> 440     1 LAKEOFTHEWOODS    1.0986     -0.6817
#> 441     0        LESUEUR    2.5337      0.1944
#> 442     1        LESUEUR    1.4586      0.1944
#> 443     0        LESUEUR    1.5261      0.1944
#> 444     0        LESUEUR    1.3863      0.1944
#> 445     0        LESUEUR    1.2238      0.1944
#> 446     0        LINCOLN    2.8679      0.4449
#> 447     0        LINCOLN    2.3702      0.4449
#> 448     0        LINCOLN    2.0794      0.4449
#> 449     1        LINCOLN    1.2809      0.4449
#> 450     0           LYON    1.8871      0.3947
#> 451     1           LYON    1.9459      0.3947
#> 452     0           LYON    1.6487      0.3947
#> 453     0           LYON    2.4932      0.3947
#> 454     0           LYON    1.6487      0.3947
#> 455     0           LYON    2.1972      0.3947
#> 456     0           LYON    1.7750      0.3947
#> 457     0           LYON    1.5476      0.3947
#> 458     0         MCLEOD    2.3418      0.1404
#> 459     0         MCLEOD    1.3863      0.1404
#> 460     0         MCLEOD    0.6419      0.1404
#> 461     0         MCLEOD    2.3026      0.1404
#> 462     0         MCLEOD    0.8755      0.1404
#> 463     0         MCLEOD    1.5041      0.1404
#> 464     0         MCLEOD    1.0647      0.1404
#> 465     1         MCLEOD    0.1823      0.1404
#> 466     0         MCLEOD    0.2624      0.1404
#> 467     1         MCLEOD    0.5306      0.1404
#> 468     1         MCLEOD    3.2387      0.1404
#> 469     1         MCLEOD   -2.3026      0.1404
#> 470     0         MCLEOD    2.3702      0.1404
#> 471     0       MAHNOMEN    1.3863      0.1496
#> 472     1       MARSHALL    0.4700      0.0138
#> 473     0       MARSHALL    3.1739      0.0138
#> 474     1       MARSHALL    0.0000      0.0138
#> 475     1       MARSHALL    0.4055      0.0138
#> 476     1       MARSHALL    0.1823      0.0138
#> 477     0       MARSHALL    1.0647      0.0138
#> 478     0       MARSHALL    3.8774      0.0138
#> 479     1       MARSHALL    0.0000      0.0138
#> 480     0       MARSHALL    2.1282      0.0138
#> 481     0         MARTIN    1.4351      0.1659
#> 482     1         MARTIN   -0.5108      0.1659
#> 483     0         MARTIN    1.9169      0.1659
#> 484     0         MARTIN    2.0281      0.1659
#> 485     0         MARTIN    2.2300      0.1659
#> 486     0         MARTIN   -0.5108      0.1659
#> 487     0         MARTIN    0.4700      0.1659
#> 488     0         MEEKER    0.8755      0.0240
#> 489     0         MEEKER    1.3863      0.0240
#> 490     0         MEEKER    1.9879      0.0240
#> 491     0         MEEKER    0.7885      0.0240
#> 492     0         MEEKER    1.1939      0.0240
#> 493     1      MILLELACS   -0.5108     -0.2101
#> 494     0      MILLELACS    1.7579     -0.2101
#> 495     0       MORRISON    0.4055     -0.0932
#> 496     0       MORRISON    0.7885     -0.0932
#> 497     0       MORRISON    1.5041     -0.0932
#> 498     0       MORRISON    0.9163     -0.0932
#> 499     0       MORRISON    1.6094     -0.0932
#> 500     1       MORRISON    1.1314     -0.0932
#> 501     0       MORRISON    1.1314     -0.0932
#> 502     0       MORRISON    1.0647     -0.0932
#> 503     0       MORRISON    1.3863     -0.0932
#> 504     0          MOWER    2.3979      0.2609
#> 505     0          MOWER    1.8718      0.2609
#> 506     0          MOWER    0.7419      0.2609
#> 507     0          MOWER    1.1314      0.2609
#> 508     0          MOWER    1.5261      0.2609
#> 509     0          MOWER    0.7885      0.2609
#> 510     0          MOWER    2.0919      0.2609
#> 511     1          MOWER    0.3365      0.2609
#> 512     0          MOWER    2.2300      0.2609
#> 513     1          MOWER    0.1823      0.2609
#> 514     0          MOWER    2.3702      0.2609
#> 515     0          MOWER    3.1822      0.2609
#> 516     0          MOWER    2.2192      0.2609
#> 517     0         MURRAY    2.5014      0.3988
#> 518     0       NICOLLET    2.1041      0.2480
#> 519     0       NICOLLET    2.3888      0.2480
#> 520     0       NICOLLET    1.4586      0.2480
#> 521     0       NICOLLET    2.7600      0.2480
#> 522     0         NOBLES    1.7047      0.4055
#> 523     0         NOBLES    1.8405      0.4055
#> 524     0         NOBLES    2.2824      0.4055
#> 525     0         NORMAN    2.1041      0.2652
#> 526     0         NORMAN    0.5306      0.2652
#> 527     1         NORMAN    0.5306      0.2652
#> 528     0        OLMSTED    1.8718      0.2432
#> 529     0        OLMSTED    1.5041      0.2432
#> 530     0        OLMSTED    2.4248      0.2432
#> 531     0        OLMSTED    2.3125      0.2432
#> 532     0        OLMSTED    1.5261      0.2432
#> 533     0        OLMSTED    2.0919      0.2432
#> 534     0        OLMSTED    0.8755      0.2432
#> 535     0        OLMSTED    1.1939      0.2432
#> 536     0        OLMSTED    1.6292      0.2432
#> 537     0        OLMSTED    1.4351      0.2432
#> 538     0        OLMSTED    0.1823      0.2432
#> 539     0        OLMSTED    0.7419      0.2432
#> 540     1        OLMSTED    0.1823      0.2432
#> 541     0        OLMSTED    1.0986      0.2432
#> 542     0        OLMSTED    0.7885      0.2432
#> 543     0        OLMSTED    2.0669      0.2432
#> 544     0        OLMSTED    1.3610      0.2432
#> 545     0        OLMSTED    0.9555      0.2432
#> 546     0        OLMSTED    1.0986      0.2432
#> 547     1        OLMSTED    0.5878      0.2432
#> 548     0        OLMSTED    0.9555      0.2432
#> 549     0        OLMSTED    2.2513      0.2432
#> 550     1        OLMSTED   -0.3567      0.2432
#> 551     0      OTTERTAIL    1.0296     -0.2047
#> 552     1      OTTERTAIL    0.1823     -0.2047
#> 553     1      OTTERTAIL    0.7885     -0.2047
#> 554     0      OTTERTAIL    2.4932     -0.2047
#> 555     1      OTTERTAIL    2.5416     -0.2047
#> 556     0      OTTERTAIL    1.1939     -0.2047
#> 557     0      OTTERTAIL    1.4586     -0.2047
#> 558     0      OTTERTAIL    1.3610     -0.2047
#> 559     1     PENNINGTON    1.3350     -0.0740
#> 560     0     PENNINGTON    1.7750     -0.0740
#> 561     1     PENNINGTON   -0.9163     -0.0740
#> 562     0           PINE    1.4351     -0.1633
#> 563     0           PINE    1.0647     -0.1633
#> 564     0           PINE    0.6931     -0.1633
#> 565     1           PINE    0.2624     -0.1633
#> 566     0           PINE    0.2624     -0.1633
#> 567     0           PINE    0.4700     -0.1633
#> 568     0      PIPESTONE    2.2513      0.4786
#> 569     0      PIPESTONE    0.5878      0.4786
#> 570     0      PIPESTONE    2.5014      0.4786
#> 571     1      PIPESTONE    1.4816      0.4786
#> 572     0           POLK    1.9459      0.2661
#> 573     1           POLK    0.4055      0.2661
#> 574     1           POLK    0.9555      0.2661
#> 575     0           POLK    2.2721      0.2661
#> 576     0           POPE    1.3610      0.2811
#> 577     0           POPE    1.2528      0.2811
#> 578     0         RAMSEY    1.9315     -0.4181
#> 579     0         RAMSEY    1.3083     -0.4181
#> 580     0         RAMSEY    0.8329     -0.4181
#> 581     0         RAMSEY    0.9933     -0.4181
#> 582     0         RAMSEY    0.7885     -0.4181
#> 583     0         RAMSEY    1.9601     -0.4181
#> 584     0         RAMSEY    0.2624     -0.4181
#> 585     0         RAMSEY    1.3610     -0.4181
#> 586     0         RAMSEY    1.2809     -0.4181
#> 587     0         RAMSEY    1.4586     -0.4181
#> 588     1         RAMSEY    0.5306     -0.4181
#> 589     1         RAMSEY    1.0647     -0.4181
#> 590     0         RAMSEY    2.1633     -0.4181
#> 591     0         RAMSEY    1.8405     -0.4181
#> 592     0         RAMSEY    1.6677     -0.4181
#> 593     0         RAMSEY    1.0296     -0.4181
#> 594     0         RAMSEY    0.2624     -0.4181
#> 595     0         RAMSEY    1.2809     -0.4181
#> 596     0         RAMSEY    1.7228     -0.4181
#> 597     1         RAMSEY    2.3224     -0.4181
#> 598     0         RAMSEY    1.7228     -0.4181
#> 599     0         RAMSEY    0.2624     -0.4181
#> 600     0         RAMSEY    1.6094     -0.4181
#> 601     0         RAMSEY    1.4110     -0.4181
#> 602     0         RAMSEY    1.2809     -0.4181
#> 603     0         RAMSEY    0.9555     -0.4181
#> 604     0         RAMSEY    0.2624     -0.4181
#> 605     0         RAMSEY    1.0296     -0.4181
#> 606     0         RAMSEY    0.5878     -0.4181
#> 607     0         RAMSEY    1.1632     -0.4181
#> 608     0         RAMSEY   -0.2231     -0.4181
#> 609     0         RAMSEY    0.0953     -0.4181
#> 610     0        REDWOOD    0.6931      0.3663
#> 611     0        REDWOOD    1.3610      0.3663
#> 612     0        REDWOOD    2.1972      0.3663
#> 613     0        REDWOOD    2.0149      0.3663
#> 614     1        REDWOOD    3.0350      0.3663
#> 615     0       RENVILLE    1.8083      0.3806
#> 616     0       RENVILLE    0.7885      0.3806
#> 617     1       RENVILLE    1.7750      0.3806
#> 618     0           RICE    2.2824      0.1931
#> 619     0           RICE    1.8718      0.1931
#> 620     0           RICE    1.5476      0.1931
#> 621     0           RICE    1.7405      0.1931
#> 622     0           RICE    2.9497      0.1931
#> 623     1           RICE    0.9163      0.1931
#> 624     0           RICE    1.1314      0.1931
#> 625     0           RICE    1.6487      0.1931
#> 626     0           RICE    2.0541      0.1931
#> 627     0           RICE    2.1041      0.1931
#> 628     0           RICE    1.5686      0.1931
#> 629     0           ROCK    2.1401      0.5280
#> 630     0           ROCK    0.5306      0.5280
#> 631     1         ROSEAU    1.8083     -0.2120
#> 632     1         ROSEAU    0.1823     -0.2120
#> 633     0         ROSEAU    2.4423     -0.2120
#> 634     1         ROSEAU    1.4816     -0.2120
#> 635     1         ROSEAU    1.3083     -0.2120
#> 636     0         ROSEAU    2.3418     -0.2120
#> 637     1         ROSEAU    1.2528     -0.2120
#> 638     0         ROSEAU    1.1632     -0.2120
#> 639     0         ROSEAU    1.3083     -0.2120
#> 640     0         ROSEAU    1.0296     -0.2120
#> 641     0         ROSEAU    1.4110     -0.2120
#> 642     1         ROSEAU    0.2624     -0.2120
#> 643     1         ROSEAU    0.5878     -0.2120
#> 644     1         ROSEAU    1.4586     -0.2120
#> 645     0        STLOUIS   -0.1054     -0.4747
#> 646     1        STLOUIS   -0.5108     -0.4747
#> 647     0        STLOUIS    0.9163     -0.4747
#> 648     0        STLOUIS    0.8755     -0.4747
#> 649     0        STLOUIS    1.5476     -0.4747
#> 650     0        STLOUIS    2.4069     -0.4747
#> 651     0        STLOUIS    2.7081     -0.4747
#> 652     0        STLOUIS    2.1633     -0.4747
#> 653     0        STLOUIS    1.5261     -0.4747
#> 654     0        STLOUIS    0.4700     -0.4747
#> 655     0        STLOUIS    1.3863     -0.4747
#> 656     0        STLOUIS    0.6419     -0.4747
#> 657     0        STLOUIS    0.5306     -0.4747
#> 658     0        STLOUIS   -0.5108     -0.4747
#> 659     1        STLOUIS   -0.6931     -0.4747
#> 660     1        STLOUIS   -0.5108     -0.4747
#> 661     0        STLOUIS    2.1748     -0.4747
#> 662     1        STLOUIS    0.5306     -0.4747
#> 663     0        STLOUIS    0.4055     -0.4747
#> 664     0        STLOUIS    2.1748     -0.4747
#> 665     0        STLOUIS    2.4159     -0.4747
#> 666     0        STLOUIS    0.4700     -0.4747
#> 667     0        STLOUIS    0.1823     -0.4747
#> 668     1        STLOUIS    0.0000     -0.4747
#> 669     0        STLOUIS   -0.2231     -0.4747
#> 670     0        STLOUIS    1.4586     -0.4747
#> 671     0        STLOUIS    1.2528     -0.4747
#> 672     1        STLOUIS    0.7885     -0.4747
#> 673     0        STLOUIS    1.0986     -0.4747
#> 674     0        STLOUIS    0.6419     -0.4747
#> 675     0        STLOUIS    0.6419     -0.4747
#> 676     0        STLOUIS    0.9163     -0.4747
#> 677     0        STLOUIS    0.5878     -0.4747
#> 678     1        STLOUIS   -0.1054     -0.4747
#> 679     0        STLOUIS    2.4681     -0.4747
#> 680     0        STLOUIS    0.6419     -0.4747
#> 681     0        STLOUIS    1.0647     -0.4747
#> 682     1        STLOUIS    1.2809     -0.4747
#> 683     0        STLOUIS    1.3083     -0.4747
#> 684     0        STLOUIS    1.2809     -0.4747
#> 685     0        STLOUIS    1.1314     -0.4747
#> 686     1        STLOUIS    1.1939     -0.4747
#> 687     0        STLOUIS    1.1632     -0.4747
#> 688     0        STLOUIS    1.2238     -0.4747
#> 689     1        STLOUIS    0.5878     -0.4747
#> 690     0        STLOUIS    1.7405     -0.4747
#> 691     0        STLOUIS    1.2528     -0.4747
#> 692     0        STLOUIS    0.4700     -0.4747
#> 693     0        STLOUIS    3.4751     -0.4747
#> 694     0        STLOUIS    0.1823     -0.4747
#> 695     0        STLOUIS    0.7885     -0.4747
#> 696     0        STLOUIS   -0.1054     -0.4747
#> 697     0        STLOUIS    0.4700     -0.4747
#> 698     0        STLOUIS    0.3365     -0.4747
#> 699     0        STLOUIS    1.1632     -0.4747
#> 700     0        STLOUIS    1.9879     -0.4747
#> 701     0        STLOUIS    0.4055     -0.4747
#> 702     0        STLOUIS    0.3365     -0.4747
#> 703     0        STLOUIS    0.4700     -0.4747
#> 704     0        STLOUIS    1.6292     -0.4747
#> 705     0        STLOUIS    0.8755     -0.4747
#> 706     0        STLOUIS    0.9163     -0.4747
#> 707     0        STLOUIS    0.2624     -0.4747
#> 708     0        STLOUIS    1.7047     -0.4747
#> 709     0        STLOUIS    0.1823     -0.4747
#> 710     0        STLOUIS    0.4055     -0.4747
#> 711     1        STLOUIS    1.9879     -0.4747
#> 712     0        STLOUIS    0.1823     -0.4747
#> 713     0        STLOUIS    1.2238     -0.4747
#> 714     0        STLOUIS    1.1939     -0.4747
#> 715     0        STLOUIS    0.4700     -0.4747
#> 716     0        STLOUIS    1.3083     -0.4747
#> 717     0        STLOUIS   -0.1054     -0.4747
#> 718     0        STLOUIS    0.5306     -0.4747
#> 719     0        STLOUIS    0.4055     -0.4747
#> 720     0        STLOUIS    1.0296     -0.4747
#> 721     0        STLOUIS    1.2238     -0.4747
#> 722     0        STLOUIS    0.0000     -0.4747
#> 723     0        STLOUIS   -0.3567     -0.4747
#> 724     0        STLOUIS    0.7419     -0.4747
#> 725     0        STLOUIS    0.6931     -0.4747
#> 726     0        STLOUIS    0.0000     -0.4747
#> 727     0        STLOUIS    1.7047     -0.4747
#> 728     0        STLOUIS    0.4700     -0.4747
#> 729     0        STLOUIS    1.1632     -0.4747
#> 730     0        STLOUIS    0.6419     -0.4747
#> 731     1        STLOUIS    0.0000     -0.4747
#> 732     0        STLOUIS    1.2238     -0.4747
#> 733     0        STLOUIS    0.5878     -0.4747
#> 734     0        STLOUIS    1.1632     -0.4747
#> 735     1        STLOUIS   -0.2231     -0.4747
#> 736     0        STLOUIS    1.4816     -0.4747
#> 737     0        STLOUIS    0.4055     -0.4747
#> 738     0        STLOUIS    0.6419     -0.4747
#> 739     0        STLOUIS    0.4700     -0.4747
#> 740     1        STLOUIS    0.8329     -0.4747
#> 741     0        STLOUIS    0.9163     -0.4747
#> 742     0        STLOUIS    1.0296     -0.4747
#> 743     0        STLOUIS    0.5878     -0.4747
#> 744     1        STLOUIS    0.1823     -0.4747
#> 745     1        STLOUIS    0.6419     -0.4747
#> 746     0        STLOUIS   -1.2040     -0.4747
#> 747     0        STLOUIS    0.8329     -0.4747
#> 748     0        STLOUIS    1.5476     -0.4747
#> 749     0        STLOUIS    0.7885     -0.4747
#> 750     0        STLOUIS    0.7419     -0.4747
#> 751     0        STLOUIS   -0.2231     -0.4747
#> 752     0        STLOUIS    1.8718     -0.4747
#> 753     0        STLOUIS    1.1314     -0.4747
#> 754     0        STLOUIS    0.7419     -0.4747
#> 755     0        STLOUIS    0.0000     -0.4747
#> 756     0        STLOUIS    1.2238     -0.4747
#> 757     0        STLOUIS    0.6419     -0.4747
#> 758     0        STLOUIS    0.6419     -0.4747
#> 759     0        STLOUIS    0.8329     -0.4747
#> 760     0        STLOUIS    1.4816     -0.4747
#> 761     1          SCOTT    2.9653      0.0631
#> 762     1          SCOTT    2.2192      0.0631
#> 763     0          SCOTT    0.7419      0.0631
#> 764     0          SCOTT    2.4423      0.0631
#> 765     0          SCOTT    2.3321      0.0631
#> 766     1          SCOTT    0.7885      0.0631
#> 767     0          SCOTT    0.2624      0.0631
#> 768     0          SCOTT    1.1939      0.0631
#> 769     1          SCOTT    0.7419      0.0631
#> 770     0          SCOTT    1.4816      0.0631
#> 771     0          SCOTT    0.8329      0.0631
#> 772     0          SCOTT    1.7047      0.0631
#> 773     0          SCOTT    3.2308      0.0631
#> 774     0      SHERBURNE    1.6487     -0.6834
#> 775     0      SHERBURNE    0.8755     -0.6834
#> 776     0      SHERBURNE    1.1939     -0.6834
#> 777     0      SHERBURNE    0.9555     -0.6834
#> 778     0      SHERBURNE    1.0647     -0.6834
#> 779     0      SHERBURNE    1.1632     -0.6834
#> 780     0      SHERBURNE    0.5306     -0.6834
#> 781     0      SHERBURNE    1.5686     -0.6834
#> 782     0         SIBLEY    1.4110      0.2372
#> 783     0         SIBLEY    1.6292      0.2372
#> 784     0         SIBLEY    0.4700      0.2372
#> 785     0         SIBLEY    1.5892      0.2372
#> 786     0        STEARNS    2.0281      0.1164
#> 787     0        STEARNS    1.8718      0.1164
#> 788     0        STEARNS    2.1282      0.1164
#> 789     0        STEARNS    0.7885      0.1164
#> 790     0        STEARNS    1.2238      0.1164
#> 791     0        STEARNS    0.3365      0.1164
#> 792     0        STEARNS    1.6292      0.1164
#> 793     1        STEARNS    0.0953      0.1164
#> 794     0        STEARNS    1.9601      0.1164
#> 795     0        STEARNS    1.7579      0.1164
#> 796     0        STEARNS    2.3224      0.1164
#> 797     0        STEARNS    1.9021      0.1164
#> 798     1        STEARNS    0.9933      0.1164
#> 799     0        STEARNS    1.2238      0.1164
#> 800     1        STEARNS    0.4700      0.1164
#> 801     0        STEARNS    1.6292      0.1164
#> 802     0        STEARNS    2.0149      0.1164
#> 803     0        STEARNS    2.6810      0.1164
#> 804     0        STEARNS    0.6419      0.1164
#> 805     0        STEARNS    2.0149      0.1164
#> 806     0        STEARNS    0.9933      0.1164
#> 807     0        STEARNS    1.3350      0.1164
#> 808     0        STEARNS    0.6931      0.1164
#> 809     1        STEARNS    0.8329      0.1164
#> 810     0        STEARNS    1.6292      0.1164
#> 811     0         STEELE    2.0015      0.2698
#> 812     0         STEELE    1.3350      0.2698
#> 813     0         STEELE    1.0986      0.2698
#> 814     0         STEELE    1.5041      0.2698
#> 815     0         STEELE    2.1401      0.2698
#> 816     0         STEELE    1.6487      0.2698
#> 817     0         STEELE    1.3083      0.2698
#> 818     0         STEELE    0.4700      0.2698
#> 819     0         STEELE    2.1633      0.2698
#> 820     0         STEELE    2.3702      0.2698
#> 821     0        STEVENS    2.0919      0.4708
#> 822     0        STEVENS    1.5261      0.4708
#> 823     0          SWIFT    1.1314      0.3160
#> 824     0          SWIFT    0.9163      0.3160
#> 825     0          SWIFT    0.4700      0.3160
#> 826     0          SWIFT    1.5892      0.3160
#> 827     0           TODD    1.9315     -0.0468
#> 828     0           TODD    0.7885     -0.0468
#> 829     1           TODD    1.8083     -0.0468
#> 830     1       TRAVERSE    1.0986      0.4976
#> 831     0       TRAVERSE    1.9169      0.4976
#> 832     0       TRAVERSE    2.9653      0.4976
#> 833     0       TRAVERSE    1.4110      0.4976
#> 834     0        WABASHA    1.7918      0.1501
#> 835     0        WABASHA    2.2083      0.1501
#> 836     0        WABASHA    2.1401      0.1501
#> 837     1        WABASHA    0.1823      0.1501
#> 838     0        WABASHA    1.1632      0.1501
#> 839     0        WABASHA    2.4510      0.1501
#> 840     0        WABASHA    2.2721      0.1501
#> 841     0         WADENA    1.0986     -0.6720
#> 842     1         WADENA   -0.2231     -0.6720
#> 843     1         WADENA    1.1939     -0.6720
#> 844     0         WADENA    1.5686     -0.6720
#> 845     0         WADENA    1.5892     -0.6720
#> 846     0         WASECA   -0.6931      0.2124
#> 847     0         WASECA    2.2407      0.2124
#> 848     0         WASECA    0.5878      0.2124
#> 849     1         WASECA    0.0000      0.2124
#> 850     0     WASHINGTON    2.3321     -0.1475
#> 851     0     WASHINGTON    2.0541     -0.1475
#> 852     0     WASHINGTON    0.8329     -0.1475
#> 853     0     WASHINGTON    1.8871     -0.1475
#> 854     0     WASHINGTON    2.5096     -0.1475
#> 855     0     WASHINGTON    1.5476     -0.1475
#> 856     0     WASHINGTON    1.8405     -0.1475
#> 857     0     WASHINGTON    1.8871     -0.1475
#> 858     0     WASHINGTON    1.0647     -0.1475
#> 859     0     WASHINGTON    0.6931     -0.1475
#> 860     1     WASHINGTON    0.2624     -0.1475
#> 861     0     WASHINGTON    0.9163     -0.1475
#> 862     0     WASHINGTON    0.0953     -0.1475
#> 863     1     WASHINGTON    0.2624     -0.1475
#> 864     0     WASHINGTON    0.5306     -0.1475
#> 865     0     WASHINGTON   -0.1054     -0.1475
#> 866     0     WASHINGTON    0.5878     -0.1475
#> 867     0     WASHINGTON    1.5686     -0.1475
#> 868     1     WASHINGTON    0.5878     -0.1475
#> 869     0     WASHINGTON    1.2238     -0.1475
#> 870     1     WASHINGTON   -0.1054     -0.1475
#> 871     0     WASHINGTON    2.2925     -0.1475
#> 872     0     WASHINGTON    1.6864     -0.1475
#> 873     0     WASHINGTON    2.1518     -0.1475
#> 874     0     WASHINGTON    0.6931     -0.1475
#> 875     0     WASHINGTON    1.9021     -0.1475
#> 876     0     WASHINGTON    1.3610     -0.1475
#> 877     0     WASHINGTON    1.7918     -0.1475
#> 878     0     WASHINGTON    1.6094     -0.1475
#> 879     1     WASHINGTON    0.9555     -0.1475
#> 880     0     WASHINGTON    2.3795     -0.1475
#> 881     0     WASHINGTON    0.9163     -0.1475
#> 882     0     WASHINGTON    0.7885     -0.1475
#> 883     0     WASHINGTON    1.5686     -0.1475
#> 884     0     WASHINGTON    1.3350     -0.1475
#> 885     0     WASHINGTON    2.6027     -0.1475
#> 886     0     WASHINGTON    1.0986     -0.1475
#> 887     0     WASHINGTON    1.4816     -0.1475
#> 888     0     WASHINGTON    1.3610     -0.1475
#> 889     0     WASHINGTON    0.6419     -0.1475
#> 890     0     WASHINGTON    0.4700     -0.1475
#> 891     0     WASHINGTON    0.6419     -0.1475
#> 892     0     WASHINGTON    0.3365     -0.1475
#> 893     0     WASHINGTON    1.9021     -0.1475
#> 894     0     WASHINGTON    3.0204     -0.1475
#> 895     0     WASHINGTON    1.8083     -0.1475
#> 896     0       WATONWAN    2.6319      0.1832
#> 897     1       WATONWAN    2.3321      0.1832
#> 898     1       WATONWAN    1.7579      0.1832
#> 899     0         WILKIN    2.2407      0.2360
#> 900     0         WINONA    1.2528      0.4632
#> 901     0         WINONA    1.4351      0.4632
#> 902     0         WINONA    2.4596      0.4632
#> 903     0         WINONA    1.9879      0.4632
#> 904     0         WINONA    1.5686      0.4632
#> 905     1         WINONA    0.6419      0.4632
#> 906     1         WINONA   -0.2231      0.4632
#> 907     0         WINONA    1.5686      0.4632
#> 908     0         WINONA    2.3321      0.4632
#> 909     0         WINONA    2.4336      0.4632
#> 910     0         WINONA    2.0412      0.4632
#> 911     0         WINONA    2.4765      0.4632
#> 912     1         WINONA   -0.5108      0.4632
#> 913     0         WRIGHT    1.9169     -0.0900
#> 914     0         WRIGHT    1.6864     -0.0900
#> 915     0         WRIGHT    1.1632     -0.0900
#> 916     0         WRIGHT    0.7885     -0.0900
#> 917     0         WRIGHT    2.0015     -0.0900
#> 918     0         WRIGHT    1.6487     -0.0900
#> 919     0         WRIGHT    0.8329     -0.0900
#> 920     1         WRIGHT    0.8755     -0.0900
#> 921     0         WRIGHT    2.7726     -0.0900
#> 922     0         WRIGHT    2.2618     -0.0900
#> 923     0         WRIGHT    1.8718     -0.0900
#> 924     0         WRIGHT    1.5261     -0.0900
#> 925     0         WRIGHT    1.6292     -0.0900
#> 926     0 YELLOWMEDICINE    1.3350      0.3553
#> 927     0 YELLOWMEDICINE    1.0986      0.3553
```

The data consist of 919  observations of radon levels of houses from 85 counties.


```r
radon_county <- radon %>%
  group_by(county) %>%
  summarise(log_radon_mean = mean(log_radon),
            log_radon_sd = sd(log_radon), 
            log_uranium = mean(log_uranium),
            n = length(county))
```


```r
ggplot() +
  geom_point(data = radon,
             mapping = aes(y = log_radon, x = fct_reorder(county, log_radon, mean))) +
  geom_point(data = radon_county,
             mapping = aes(x = fct_reorder(county, log_radon_mean), y = log_radon_mean),
             colour = "red") +
  coord_flip() + 
  labs(y = "log(radon)", x = "")
```

<img src="multilevel_files/figure-html/unnamed-chunk-5-1.png" width="70%" style="display: block; margin: auto;" />

Relationship between mean and sample size

```r
ggplot(radon_county, aes(y = log_radon_mean, x = log2(n))) + 
  geom_point()
```

<img src="multilevel_files/figure-html/unnamed-chunk-6-1.png" width="70%" style="display: block; margin: auto;" />

### Varying Intercepts Models

Consider the general model with an intercept for each county representing the baseline average of the county:
$$
\begin{aligned}
y_i &\sim  N(\mu_i, \sigma^2) \\
\mu_i &= \alpha_{j[i]} + \beta x_i
\end{aligned}
$$
where $j[i]$ means that observation $i$ is in county $j \in (1, \dots, 85)$.

In this particular example, $y = \mathtt{log_radon}$ and $x = \mathtt{basement}$.
$$
\begin{aligned}
\mathtt{log\_radon}_i &\sim  N(\mu_i, \sigma^2) \\
\mu_i &= \alpha_{j[i]} + \beta~\mathtt{basement}_i
\end{aligned}
$$

We can put a prior distribution on $\alpha_{j[i]}$,
$$
\begin{aligned}[t]
\alpha_{j} &\sim N(\gamma, \tau) & \text{for $i \in (1, \dots, 85)$}
\end{aligned}
$$
This parameterization nests common cases,

*Complete pooling:* When $\tau \to 0$, the intercepts are the same,
$$
\begin{aligned}[t]
\alpha_j &= \gamma  & \text{for all $j$.}
\end{aligned}
$$

*No pooling:* When $\tau \to \infty$, prior distribution on the intercepts is equivalent to an improper normal distribution, and there is no shrinkage,
$$
p(\alpha_j) \propto 1,
$$
for all $j$.

*Partial pooling:* When $\tau$ is a parameter, the amount of shrinkage can be estimated from the data.
A common prior is $\tau \sim N(0, 2.5)$,
$$
\tau \sim N^{+}(0, 2.5) .
$$

The partial pooling model 


### Varying Intercept Model


### Varying Slope Model


$$
\begin{aligned}
\mathtt{log\_radon}_i &\sim  N(\mu_i, \sigma^2) \\
\mu_i &= \alpha_{j[i]} + \beta_{j[i]}~\mathtt{basement}_i
\end{aligned}
$$


### Group Level Predictors

The `radon` dataset also contains the county-level measurements of `uranium`.

One way to include county level measurements is to model the county-level intercepts. 
The values of each county intercept is a function of the county-level uranium.
$$
\begin{aligned}
\mathtt{log\_radon}_i &\sim  N(\mu_i, \sigma^2) \\
\mu_i &= \alpha_{j[i]} + \beta_{j[i]}~\mathtt{basement}_i
\alpha_{j} \sim  N(\gamma_0 + \gamma_1~\mathtt{log\_uranium}_j, \tau)
\end{aligned}
$$


Alternatively, we can model model the county-level intercepts. 
The values of each county intercept is a function of the county-level uranium.
$$
\begin{aligned}
\mathtt{log\_radon}_i &\sim  N(\mu_i, \sigma^2) \\
\mu_i &= \alpha_{j[i]} + \beta_{j[i]}~\mathtt{basement}_i \\
\alpha_{j} &\sim  N(\gamma_0 + \gamma_1~\mathtt{log\_uranium}_j, \tau)
\end{aligned}
$$

### lme4

In R, the most widely used package to estimate mixed-effects models is **lme4**. 
This esimates models using maximum likelihood or restricted maximum likelihood methods (REML). 
This will be faster than using full-Bayesian methods but also underestimate the uncertainty, as well as being a worse approximation of the posterior.
Additionally, in frequentist inference, the meaning of the random effects is different; they are nuisance parameters and not given standard errors.

See @Bates2010a and @BatesMaechlerBolkerEtAl2014a for introductions to mixed-effects models with **lme4**.
These are also good introductions to classical approaches to mixed effects models.


```r
library("lme4")
#> Loading required package: Matrix
#> 
#> Attaching package: 'Matrix'
#> The following object is masked from 'package:tidyr':
#> 
#>     expand
```

Complete pooling

```r
fit_pooled <- lm(log_radon ~ county + floor, data = radon)
```
County-varying intercepts with no-pooling

```r
fit_intercept_nopool <- lm(log_radon ~ floor, data = radon)
```
County-varying intercepts with partial-pooling

```r
fit_intercept_partial <- lmer(log_radon ~ (1 | county) + floor, data = radon)
```
Varying slopes with no pooling:

```r
fit_slope_nopool <- lm(log_radon ~ county * floor, data = radon)
```
Varying slopes with partial pooling:

```r
fit_slope_partial <- lmer(log_radon ~ (1 + floor | county), data = radon)
```

Including a county-level variable (`log_uranium`) in various models:

With no-pooling,

```r
fit_slope_partial <- lm(log_radon ~ floor + log_uranium, data = radon)
```
With varying-intercepts

```r
fit_slope_partial <- lmer(log_radon ~ (1 | county) + floor + log_uranium, data = radon)
```
With varying-intercepts and slopes,

```r
fit_slope_partial <- lmer(log_radon ~ (1 + floor | county) +  log_uranium, data = radon)
```

### rstanarm

Some multilevel models can also be estimated using the **rstanarm** functions `stan_glmer` and `stan_lmer`.
These functions have syntax similar to **lme4** functions, but estimate the mixed models using Bayesian methods with Stan.

Complete pooling

```r
fit_pooled <- stan_glm(log_radon ~ county + floor, data = radon)
```
County-varying intercepts with no-pooling

```r
fit_intercept_nopool <- stan_glm(log_radon ~ floor, data = radon)
```
County-varying intercepts with partial-pooling

```r
fit_intercept_partial <- stan_glmer(log_radon ~ (1 | county) + floor, data = radon)
```
Varying slopes with no pooling. *There is an error estimating this*

```r
fit_slope_nopool <- stan_glm(log_radon ~ -1 + county + county:floor, data = radon,
                             prior = normal(scale = 1))
```
Varying slopes with partial pooling:

```r
fit_slope_partial <- stan_glmer(log_radon ~ (1 + floor | county), data = radon)
```

Including a county-level variable (`log_uranium`) in various models:

With no-pooling,

```r
fit_slope_partial <- stan_glm(log_radon ~ floor + log_uranium, data = radon)
```
With varying-intercepts

```r
fit_slope_partial <- stan_glmer(log_radon ~ (1 | county) + floor + log_uranium, data = radon)
```
With varying-intercepts and slopes,

```r
fit_slope_partial <- stan_glmer(log_radon ~ (1 + floor | county) +  log_uranium, data = radon)
```

## Pooling of Hierarchical Parameters

This is easiest understood in the case of a model of group means,
$$
\begin{aligned}[t]
y &\sim \dnorm(\mu_{j[i]}, \sigma^2) \\
\mu_{j} &\sim \dnorm(\gamma, \tau^2) .
\end{aligned}
$$
Each group has size $n_j$.

Sample size, $n_j$               Estimate of $\hat{\mu}_j$
-------------------------------- ---------------------------------------------------------------
$n_j = 0$                        $\hat{\mu}_j = \gamma$ (complete pooling)
$n_j < \frac{\sigma^2}{\tau^2}$  $\hat{\mu}_j$ closer to $\gamma$
$n_j = \frac{\sigma^2}{\tau^2}$  $\hat{\mu}_j = \frac{1}{2} \bar{y}_j + \frac{1}{2} \gamma$
$n_j > \frac{\sigma^2}{\tau^2}$  $\hat{\mu}_j$ closer to $\bar{y}_j$
$n_j = \infty$                   $\hat{\mu}_j = \bar{y}_j$ (no pooling)

If the hyperparameters were known, the posterior of $\mu_j$ is
$$
\mu_j | y, \gamma, \sigma, \tau \sim \dnorm(\hat{\mu}_j, V_j)
$$
where
$$
\begin{aligned}[t]
\hat{\mu}_j &= \frac{\frac{n_j}{\sigma^2} \bar{y}_j + \frac{1}{\tau^2} \gamma}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}} \\
V_j &= \frac{1}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}}
\end{aligned}
$$

Some crude estimates given $\mu_j$.

The *data variance*, $\sigma^2$, is the residual variance,
$$
\E(\sigma^2 | y, \mu)  = \frac{1}{n} \sum_{i = 1}^n (y - \mu_{j[i]})^2 .
$$
The global mean is approximately the average of the group-level means,
$$
\begin{aligned}[t]
\E(\gamma | y, \mu) &= \frac{1}{J} \sum_{i = 1}^n \mu_j \\
\Var(\gamma | y, \mu) &= \frac{1}{J} \tau^2
\end{aligned}
$$
The group level variance is $\tau^2$ is,
$$
\E(\tau^ | y, \mu) = \frac{1}{J} \sum_{j = 1}^J (\mu_j - \gamma)^2
$$


## Extensions

- Including group-level covariates
- Prior distributions
- Prediction

    - new obs in existing groups
    - new group
    - new obs in new group
    
- Modeling correlation between intercept and slopes
- Non-nested models

## References 

Texts:

- @GelmanHill2007a [Ch. 11-17].
- @BDA3 [Ch 5] "Hierarchical Models"
- @BDA3 [Ch 15] "Hierarchical Linear Models"

Other

- Stan models for [ARM](https://github.com/stan-dev/example-models/wiki/ARM-Models)
- http://mc-stan.org/documentation/case-studies/radon.html
- https://biologyforfun.wordpress.com/2016/12/08/crossed-and-nested-hierarchical-models-with-stan-and-r/
