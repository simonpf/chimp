# Training the retrieval model

The next step in building a retrieval is to set up a neural network model
and training scheme and use them to train the retrieval. For this basic
example, we will only consider a retrieval that ingests observations from
multiple satellite but only a single time-step at once.

```{tip}
During training CHIMP produces a range of artifacts such as input and output data statistics, training checkpoint and logs, and the model file. To keep retrievals well-organized it is recommended to perform the training for each model in separate, designated folder.
```

## Defining the model and training schedule

The principal ingredients to training a CHIMP retrieval are a definition of the retrieval model and the training schedule used to train this model. Both model and training schedule are defined using configuration files in [toml](https://toml.io/en/)
format.

By default, the CHIMP commands for exploratory data analysis, learning rate search, and training expect the model and training configuration files to be in files named ``model.toml`` and ``training.toml`` in the current working directory. However, all commands also provide options ``--model_config`` and ``--training_config`` to customize the location of these files.

### The model configuration file

The model configuration file specifies the architecture of the underlying neural network model as well as the inputs and outputs of the retrieval. The architecture
is defined in a table ``[architecture]``. The example file below defines an ``EncoderDecoder`` architecture using the ``EfficientNetV2-S`` preset.

Inputs are defined as subtables of the ``[input]`` table using
``[input.<input_name]``. Each input table must specify the number of features in
the input (``n_features``) and, if desired, a normalization scheme for the
input.

Retrieval outputs are defined in a similar fashion using ``[output.<output_name>]``.
The output should defines the type of the output, here 32 quantiles (``Quantiles``) of the posterior distribution using quantile regression.

```
name = "goes_gpm"

[architecture]
name = "EncoderDecoder"
preset = "EfficientNetV2-S"

[input.goes]
n_features = 16
normalize = "minmax"

[input.gmi]
n_features = 13
normalize = "minmax"

[input.atms]
n_features = 9
normalize = "minmax"

[output.surface_precip]
kind = "Quantiles"
quantiles = 32
```

### The training configuration file

The training configuration file defines the training regime used to train the
retrieval. The training is organized into consecutive stages each of which can
have a  different configuration.

Every stage must specify the base paths of the training and validation data
(``training_data_path`` and ``reference_data_path``, respectively) as well as
the names of the input and reference datasets from which to load the training
data.

Additional required configuration settings are the optimizer, learning-rate scheduler,
batch_size and the metrics to log.

The file below defines two training epochs. A ``warmup`` epoch which uses only a tenth of the training data (``sample_rate = 0.1``) and uses a ``Warmup`` scheduler, which linearly increases the learning rate to the target learning rate specified in ``optimizer_args``.

```toml
[warmup]
training_data_path = "/path/to/training_data"
input_datasets = ["goes_16", "atms", "gmi"]
reference_datasets = ["mrms"]
n_epochs = 20
optimizer = "AdamW"
optimizer_args = {lr = 1e-3}
scheduler = "Warmup"
batch_size = 32
sample_rate = 0.1
metrics = ["Bias", "MSE", "CorrelationCoef", "PlotSamples"]

[stage_1]
training_data_path = "/path/to/training_data"
validation_data_path = "/path/to/validation_data"
input_datasets = ["goes_16", "atms", "gmi"]
reference_datasets = ["mrms"]
n_epochs = 20
optimizer = "AdamW"
optimizer_args = {lr = 1e-3}
scheduler = "CosineAnnealingLR"
scheduler_args = {T_max = 20}
batch_size = 32
metrics = ["Bias", "MSE", "CorrelationCoef", "PlotSamples"]
reuse_optimizer = true
```

## Exploratory data analysis

Prior to the actual model training it is recommended and, sometimes, required to perform an exploratory data analysis (EDA) using CHIMP. The EDA iterates twice over the dataset recording basic statistics of all input and output variables. The statistics are useful as a sanity check for the training data and are used by CHIMP so set the input normalization coefficients in the retrieval model.

The command to perform the EDA is

```
chimp eda
```

After a successful EDA run, the working directory will contain a new ``stats`` directory, which contains the the calculated statistics as NetCDF4 files.

## Running the training

Finally, to run the training, all that is require is to call the CHIMP ``train`` command:

```
chimp train
```
