# Evaluation

Following successful training, the next step of retrieval development is
typically assing the accuracy of the newly trained model. CHIMP provides
built-in testing functionality aiming to provide a rapid way of calculating
relevant accuracy metrics.

## Model testing

The ``chimp test`` command implements CHIMP's model testing functionality. Assuming the retrieval model ``retrieval_model.pt`` was trained on ``goes_16``, ``atms`` and ``gmi`` input datasets and the ``mrms`` reference dataset, the resulting model can be evaluated on a separate test set using the command:

````shell
chimp test retrieval_model.pt /path/to/test_data results.nc --input_datasets goes16,atms,gmi --reference_datasets mrms --tile_size 256 --batch_size 32
````

This command will iterate over the test data found in ``/path/to/test_data`` and
calculate bias, mean-squared error and the linear correlation coefficient for
all scalar retrieval targets. The accuracy of all predictions will be assessed for all pixels as well
as conditional on the availability of each input. The variables will be stored in NetCDF4 format containing
the results of the retrieval assessment stored in variables following the naming pattern ``<output_name>_<metric_name>`` for metrics calculated over all valid pixels and ``<output_name>_<metric_name>_<input_name>`` for metrics calculated conditional on the availability of input ``<input_name>``.
