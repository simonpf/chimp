# CHIMP

CHIMP, the Chalmers/CSU integrated multi-satellite retrieval platform, is a Python frame work for building satellite retrieval and forecast systems.


## Installation

The currently recommended way to install CHIMP is by cloning the git repository and using the provided ``chimp.yml`` conda environment file to install the required dependencies.

### Cloning the git repository

```
git clone https://github.com/simonpf/chimp
```

### Creating the conda environment

```
cd chimp
conda env create --file chimp.yml
```

### Installing CHIMP

```
pip install -e .
```
