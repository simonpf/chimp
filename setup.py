from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version = {}
exec(open("chimp/version.py", "r").read(), version)

setup(
    name="chimp",
    version=version["__version__"],
    description="The Chalmers Integrated Multi-Satellite Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonpf/chimp",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=[
        "click",
        "rich",
        "pytorch",
        "pytorch-lightning",
        "numpy",
        "scipy",
        "xarray",
        "pandas",
        "quantnn",
        "tensorboard",
        "dask",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    project_urls={
        "Source": "https://github.com/simonpf/chimp/",
    },
    entry_points={
        "console_scripts": ["chimp=chimp.cli:chimp"],
    },
    include_package_data=True,
    package_data={"chimp": ["areas/*.yml"]},
)
