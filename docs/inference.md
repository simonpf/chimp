# Inference

The aim of all retrieval development is, of course, the applications of the
retrieval to actual satellite observations. To ease the transition from model
development to application, CHIMP also provide basic inference capability.


## Running CHIMP retrievals

CHIMP retrievals can be run using the ``chimp process`` command. The ``process``
command assumes that the input observations were prepared in the same way as the
training data, ideally using CHIMP's extract data command. The ``process``
command processes all input files in a given folder. The retrieval is run on the
full domain by splitting it up into tiles and assembling the results.

The CHIMP model ``retrieval_model.pt`` trained on input datasets ``goes_16``, ``atms``, and ``gmi`` can be run as follows:

````
chimp process -v retrieval_model.pt goes16 atms gmi path/to/test_data/  results/ --tile_size 256 --overlap 64
````

This command will process all input data files using a tile size of 256 pixels at the base scale
and a tile-overlap of 64. The output is stored in NetCDF4 format in the ``results`` directory.
