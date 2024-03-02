# Extracting the training data

As with every machine-learning project, the first step in building satellite
retrieval with CHIMP consists of extracting the training data. CHIMP natively
supports a range of satellite observations and provides a convenient command
line interface to extract training data from them.

## Data organization

CHIMP organizes the input and reference data used for the retrieval training into separate different folders. Within each folder training samples are organized by sample time. In order to allow combining observations with slightly diffent observations time, the sample times are rounded to a given time step. For a retrieval using a time step of 15 minutes, for example, files for minutes 0, 15, 30, and 45 or every hour are create.

The example below shows the directory structure for training data for a retrieval
trained to estimate precipitation from GOES observations based on reference data
from ground-based precipitation radars provided by the
Multi-Radar/Multi-Sensor System (MRMS).

```
├── goes
│   ├── goes_20230601_00_00.nc
│   ├── goes_20230601_00_15.nc
│   ├── goes_20230601_00_30.nc
│   ├── goes_20230601_00_45.nc
│   └── ...
└── mrms
    ├── mrms_20230601_00_00.nc
    ├── mrms_20230601_00_15.nc
    ├── mrms_20230601_00_30.nc
    ├── mrms_20230601_00_45.nc
    └── ...

```

## Setting up ``pansat``

CHIMP uses the ``pansat`` package to perform cached downloads of the satellite and reference data from which the training data is derived. If not explicitly configured, ``pansat`` will download satellite data into the current working directory, which may not be ideal. This behavior can be changed by registering a data directory with ``pansat``. This can be done using:

```
pansat registry add data_directory <name> <path>
```

To check that ``pansat`` has been set up successfully, run
```
pansat registry list
```
and the newly added data directory should be listed in the output.

> **NOTE:** The pansat settings are stored in a configuration file in the user directory. If the data in the user directory does not persist between session, the command adding the data directory must be repeated at the beginning of every session.

### Adding credentials for NASA GES DISC data

To be able to download data from NASA GES DISC servers for you, a valid username and password needs to be added to ``pansat``. This can be done using:

```shell
pansat account add "GES DISC" <user name>

```
If this is your first time adding credentials to ``pansat``, ``pansat`` will ask you to setup a password. ``pansat`` will query for this password whenever it needs access to the GES DISC or any other credentials. 

> **NOTE**: ``pansat`` stores the password in an encrypted identity file in the pansat configuration directory (``$HOME/.config/pansat`` by default). Although the file is encrypted, it is recommended to use a throw-away password to store in ``pansat``.

You can run the following command to verify that the correct account has been added to pansat:

```
panst account list-accounts
```

## Synopsis

All training-data preparation tasks are performed using a single command, which takes the general form:

```
chimp extract_data <dataset_name> <year> <month> <day_1 day_2> <output_path> --domain <domain_name> --time_step --n_processes N
```

Where
- ``dataset`` is the name of the dataset that the data should be extracted from,
- ``year`` is an integer specifying the year for which to extract data,
- ``month`` is an integer specifying the month for which to extract data or '?' to extract data for the full year,
- ``day_1 day2`` is an optional list of days for which to extract data. If omitted, data for the full month will be extracted,
- ``output_path`` is a path pointing to the directory to which to write the extracted training, validation or test samples,
- ``domain`` specifies the name of a predefined spatial domain,
- ``time_step`` is the retrieval time step in minutes,
- ``N`` specifies the number of processes to use to extract the data.


### Extracting GOES observations

Following this pattern, training data from GOES-16 observations for July 2023 can be extracted using

```
chimp extract_data goes_16 2023 7 /path/to/training_data --domain conus_plus --n_processes 4
```

Note that, due to the size of the GOES observations, this process likely takes relatively long.


### Extracting GMI observations

```
chimp extract_data gmi 2023 7 /path/to/training_data --domain conus_plus --n_processes 4
```

### Extracting ATMS observations

```
chimp extract_data atms 2023 7 /path/to/training_data --domain conus_plus --n_processes 4
```

### Extracting MRMS precipitation estimates

```
chimp extract_data mrms 2023 7 /path/to/training_data --domain conus_plus --n_processes 4
```

### Extracting GPM CMB precipitation estimates

```
chimp extract_data cmb 2023 7 /path/to/training_data --domain conus_plus --n_processes 4
```
