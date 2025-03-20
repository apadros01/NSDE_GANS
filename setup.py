import setuptools
from pathlib import Path

this_directory = Path(__file__).parent

setuptools.setup(
    name = 'nsde_gans',
    version = '1.0.0',
    author = 'Àlex Padrós',
    description = 'Generate synthetic tabular data with the use of neural stochastic differential equations as generative adversarial networks',
    packages = setuptools.find_packages(), #find all packages which can be found by __init__.py
    install_requires = []
)