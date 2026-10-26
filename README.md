# Oscillon
<<<<<<< HEAD

<p align="center">
  <img width="50%" alt="Image" src="https://github.com/user-attachments/assets/769beeef-268a-46e7-bfed-89a6ee60d541" />
</p>

=======
<p align="center">
  <img width=50% alt="Image" src="https://github.com/user-attachments/assets/769beeef-268a-46e7-bfed-89a6ee60d541" />
</p>
>>>>>>> main
[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/lburtle/oscillon/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/lburtle/oscillon/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/oscillon
[conda-link]:               https://github.com/conda-forge/oscillon-feedstock
[pypi-link]:                https://pypi.org/project/oscillon/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/oscillon
[pypi-version]:             https://img.shields.io/pypi/v/oscillon?include_prereleases
[rtd-badge]:                https://readthedocs.org/projects/oscillon/badge/?version=latest
[rtd-link]:                 https://oscillon.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/lburtle/oscillon/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/lburtle/oscillon

<!-- prettier-ignore-end -->

This package is intended to be a machine learning implementation and tool for Combinatorial Threshold Linear Networks (CTLNs). The name is inspired
by the oscillating and dynamic behavior of oscillons, similar to the type of
behavior exhibity in CTLNs. This package/repository is currently under
development. Thank you for your patience.

Desired usage:

```
import oscillon
from oscillon import cycle

N = 3
eps = 0.1
delta = 0.5
init = "sparse"

model = cycle(N, eps, delta, sparse)    # wrap in W0, th0 = p2m(params0)

## spec wrapped into model init

# OR

model = cycle(N)    # default eps, delta, init

targets = ...

## Targets need x0 (or defined separately), t (time steps)
## Maybe possible for targets to contain all necessary info


model.train(targets, n_steps, warm, cryst)  # or maybe soft, hard

## Make sure to wrap R, b readout in train loop

xs = simulate(model, n_steps)[burn_in:]     # model contains W, theta

```
