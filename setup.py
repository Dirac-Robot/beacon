from setuptools import setup, find_packages
from beacon import __version__

setup(
    name='beacon',
    version=__version__,
    packages=find_packages(),
    install_requires=['rethinkdb', 'pandas', 'tqdm', 'pytz', 'loguru', 'numpy', 'setuptools', 'pyyaml']
)
