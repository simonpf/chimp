# CHIMP

CHIMP, *the Chalmers/CSU integrated multi-satellite retrieval platform**, is a Python frame work for building satellite retrieval and forecast systems.


## Installation

The currently recommended way to install CHIMP is by cloning the git repository and using the provided ``chimp.yml`` conda environment file to install the required dependencies.

### Cloning the git repository

The most recent version of the CHIMP source code can be obtained by cloning the repository using ``git``:

```
git clone https://github.com/simonpf/chimp
```

### Creating the conda environment

The ``conda.yml`` file in the ``chimp`` directory defines a conda environment with all of CHIMP's software requirements.

```
cd chimp
conda env create --file chimp.yml
conda activate chimp
```

### Installing CHIMP

Finally, use ``pip`` to install CHIMP:

```
pip install -e .
```

After successful installation the ``chimp`` command should be available and invoking it should produce the following output on the command line:

```shell
$ chimp
Usage: chimp [OPTIONS] COMMAND [ARGS]...

  CHIMP: The CSU/Chalmers integrated multi-satellite retrieval platform.

Options:
  --help  Show this message and exit.

Commands:
  eda           Perform exploratory data analysis (EDA).
  extract_data  Extract training, validation, and test data.
  lr_search     Perform learning-rate (LR) search .
  process       Process input files.
  train         Train a retrieval model.
```
