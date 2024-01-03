## Multivariate Hawkes processes for sparse estimation of neural networks intensities
Code for my semester project "Multivariate Hawkes processes for sparse estimation of neural networks intensities", carried out at the Audiovisual Communications Laboratory at EPFL in the Fall 2023 semester.

The available code implements the computation of the log-likelihood of a multivariate Hawkes process, given
a realization. Minimization of the negative log-likelihood is implemented with the Polyatomic Frank-Wolfe algorithm, as presented in [this paper](https://ieeexplore.ieee.org/document/9707878), with modifications made for this particular Hawkes process problem.

Implementation was done in Python, using the [Pyxu](https://pyxu-org.github.io/index.html) library.
