# Documentation for quadrature

The function `expectation_2d` calculates expectations $g(f)$, where $f$ is bivariate normal with mean $\mu$ and covariance $\Sigma$. This document explains how.

#### One-dimensional case

The function `np.polynomial.hermite.hermgauss` produces quadrature points and weights to approximate the integral:

$t(g) = \int g(x) \exp(-x^2) dx$

Consider the expectation of the function $g$ under a normal distribution with mean $0$ and variance $\frac{1}{2}$. Its PDF will be:

$p(x) = \frac{1}{\sqrt{\pi}} \exp (-x^2)$

We thus see that:

$\int g(x) p(x) dx = \frac{1}{\sqrt{\pi}} t$

In general, we are interested in computing $\mathbb{E}[g(X)]$, where $X \sim \mathcal{N}(\mu, \sigma^2)$. We can express the random variable $X$ as $X = \sqrt{2} \sigma Z + \mu$, where $Z \sim \mathcal{N}(0, \frac{1}{2})$. Hence:

$\mathbb{E} [g(X)] = \mathbb{E} [g(\sqrt{2} \sigma Z + \mu)] = \int g(2 \sigma z + \mu) p(z) dz = \frac{1}{\sqrt{\pi}} \int g(\sqrt{2} \sigma z + \mu) \exp(-z^2) dz$

which can now be evaluated using the default quadrature.

#### Two-dimensional case

In the two-dimensional case, we note that if $X \sim \mathcal{N}(\mu, \Sigma)$, then $X = \sqrt{2} L Z + \mu$, where $L L^\intercal = \Sigma$ is a matrix square root of $\Sigma$ and $Z \sim \mathcal{N}(\mathbf{0}, \frac{1}{2} \mathbf{I})$. This means that we can form an integration grid by considering the Cartesian product of the 1D locations. The weights are given by the outer product of the 1D weights. Finally, the pre-factor becomes $\frac{1}{\pi}$.