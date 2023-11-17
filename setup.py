from setuptools import setup, find_packages


install_requires = [
    "jaxlib",
    "jax",
    "jaxtyping",
    "jax_dataclasses",
    "matplotlib",
    "pandas",
    "jupyter",
    "ipykernel",
    "tqdm==4.50",
    "scikit-learn",
    "nvidia-cublas-cu11==11.11.3.6",
    "nvidia-cuda-nvcc-cu11==11.8.89",
    "nvidia-cuda-runtime-cu11==11.8.89",
    "nvidia-cudnn-cu11==8.9.1.23",
    "nvidia-cufft-cu11==10.9.0.58",
    "nvidia-cusolver-cu11==11.4.1.48",
    "nvidia-cusparse-cu11==11.7.5.86",
    "ipympl"
]


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Koopman-Kernel-Regression",
    version="0.1.0",
    description="Functionalities for Koopman-Kernel-Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
    zip_safe=False,
)
