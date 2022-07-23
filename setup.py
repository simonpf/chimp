from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version = {}
exec(open("cimr/version.py", "r").read(), version)

setup(
    name="cimr",
    version=version["__version__"],
    description="The Chalmers Integrated Multi-Satellite Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonpf/cimr",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=[
        "torch",
        "pytorch-lightning",
        "numpy",
        "scipy",
        "xarray",
        "pandas",
        "quantnn>=0.0.4dev"
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    project_urls={
        "Source": "https://github.com/simonpf/cimr/",
    },
    entry_points = {
        'console_scripts': ['cimr=cimr.bin:cimr'],
    },
)
