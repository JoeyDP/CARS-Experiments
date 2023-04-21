from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "matplotlib",
        "joblib",
        "sklearn",
        "numba",
        "implicit",
    ],
    entry_points={},
)
