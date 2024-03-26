# Inference

The aim of all retrieval development is, of course, the applications of the
retrieval to actual satellite observations. The ease the transition from model
development to application, CHIMP also provide limited inference capability.


## Running CHIMP retrievals

CHIMP retrievals can be run using the ``chimp process`` command. The ``process`` command assumes that data was prepared in the same way as the training data. In contrast to training, however, ``process`` will process data from the full domain by splitting it up into tiles as assembling the results.

The CHIMP model ``retrieval_model.pt`` trained on input datasets ``goes16``, ``atms``, and ``gmi`` can be run as follows:

````
chimp process -v gpm_cpcir.pt cpcir atms path/to/test_data/  results/ --tile_size 256
````

The command will process all input data files and produce corresponding output
files in NetCDF4 format in the ``results`` directory.
