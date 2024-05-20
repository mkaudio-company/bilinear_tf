# bilinear_tf

This crate implements Cohen's class of time-frequency distributions in Rust. It allows computation of the time-frequency distribution of a signal with a chosen kernel function applied in the ambiguity domain.

## Overview

Cohen's class distributions provide a general framework for time-frequency analysis of signals. They are defined in terms of a bilinear transformation of the signal's ambiguity function, allowing for flexible and adaptive time-frequency representations.

## Features

- Compute the ambiguity function of a signal.
- Apply a kernel function to the ambiguity function.
- Transform the modified ambiguity function back to the time-frequency domain.
- Use different kernel functions (e.g., Wigner, Choi-Williams, Rihaczek, and Zhao-Atlas-Marks).

## Changelog

* 0.2.0 - Added rayon crate for multithreading purpose.
* 0.1.0 - 0.1.1 - Initial release.

# Citation
Cohen, Leon. Time-Frequency Analysis. Prentice Hall, 1995.

Choi, H., & Williams, W. J. "Improved time-frequency representation of multicomponent signals using exponential kernels." IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37, no. 6, June 1989, pp. 862–871.

Rihaczek, Jan. "Waveforms Correlated with Quadratic Forms of Themselves." IRE Transactions on Information Theory, vol. 12, no. 3, 1966, pp. 225-231. doi:10.1109/TIT.1966.1053887.

Zhao, Y., Atlas, L. E., & Marks, R. J. "The use of cone-shape kernels for generalized time-frequency representations of nonstationary signals." IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 38, no. 7, July 1990, pp. 1084–1091.
