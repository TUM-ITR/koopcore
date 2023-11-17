<p align = "left">
  <img src="misc/finalLogo.svg" alt="SVG Image" style="width:80%;"/>
</p>

# `koopcore`: Koopman Kernels for Learning Dynamical Systems from Trajectory Data
`koopcore` is a Python library designed for learning linear time-invariant (LTI) predictors of dynamical systems. This library provides tools for fitting linear predictors with estimators based on Koopman Kernels.

Please note that `koopcore` is currently under highly active development and some parts might still be a work in progress as we continuously add new features and improvements.

## Installation
Create an environment and install koopcore as a local package.
```
python -m venv koopcore_env
source koopcore_env/bin/activate
pip install --find-links "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" -e .
```
## Experiments
Jupyter notebooks containing the experiments are placed in the [`experiments`](./experiments) directory.
## References
This repository contains an implementation of the paper:

[1] Bevanda, P., Beier, M., Lederer, A., Sosnowski, S., Hüllermeier E., & Hirche, S. "Koopman Kernel Regression" in *Advances
in Neural Information Processing Systems*, 2023 [[arxiv]](https://arxiv.org/abs/2305.16215)

---


If you found this software useful for your research, consider citing us.
```
@inproceedings{KKR_neurips2023,
  title = {Koopman Kernel Regression},
  author = {Bevanda, Petar and Beier, Max and Lederer, Armin and Sosnowski, Stefan and H{\"u}llermeier, Eyke and Hirche, Sandra},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2023},
  eprint = {2305.16215},
}
```
