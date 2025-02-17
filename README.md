# Basic MLE

This project is basic MLE example, which generates data by loading Iris dataset, trains a simple neural network and infers using the trained model.

## Project structure

This project has a modular structure, where each folder has a specific duty. Note that not all folders and files are present initially but they will be added (or in some cases should be added manually).

```
MLE_BASIC_EXAMPLE
├── data                            # Data files used for training and inference
│   ├── inference_data.csv          # Inference features
│   ├── X_train_data.csv            # Training features
│   └── y_train_data.csv            # Training targets
├── data_process
│   ├── __init__.py
│   └── data_generation.py          # Script used for data loading and saving
├── inference
│   ├── __init__.py
│   ├── Dockerfile
│   └── run.py                      # Script used for inference
├── models                          # Folder used to store models
│   └── ...
├── results                         # Folder used to store inference results  
│   └── ...
├── training
│   ├── __init__.py
│   ├── Dockerfile
│   └── train.py                    # Script used for model training
├── __init__.py
├── .gitignore
├── README.md
├── requirements.txt                # Files with module requirements
├── settings.json                   # All configurable parameters and settings
├── unittests.py                    # Test cases for data generation and training scripts
└── utils.py                        # Utility functions and classes that are used in scripts
```

## Working with project

### Data loading

Data should already be present in `data` folder (that folder contains three files: `inference_data.csv`, `X_train_data.csv` and `y_train_data.csv`). But if there are problems with dataset, they can be generated using this command:

```
python3 data_process/data_generation.py
```

### Training process

Training process can be done in two ways: on a local machine or using Docker.

#### Train model locally

To train model locally you have to run this command in the command line:

```
python3 training/train.py
```

This command has some arguments that can be used to tune the model:

```
--hidden_neurons          # Number of neurons in each hidden layer (integers separated by dashes, e.g. 8-4)
--batch_size              # Batch size used during training
--epochs                  # Number of epochs used during training
--verbose_interval        # Interval between each epoch log (where -1 means no epoch log at all)
```

After running the command, the model will be saved in the `models` folder under a name that will appear on one of log messages that looks like this:

```
INFO - Model saved as <name_of_the_model>
```

#### Train model using Docker

Firstly, you have to build a Docker image, using this command:

```
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```

Note, that Docker daemon must be running for this command to work. Also, the building may take about 10 minutes and 7 GB of disk space.

When the image finishes building, you need to make a container by running this command (you can also use additional arguments to tune model):

```
docker run -it training_image python3 training/train.py [--hidden_neurons HIDDEN_NEURONS] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--verbose_interval VERBOSE_INTERVAL]
```

This command will start a container and make a model. To save it on a local machine, you first need to find out the container id. You can find the container id using Docker Desktop. Open the Docker Desktop, click on 'Containers', find the container and copy its id.

After running the command, look for log message that contains model name:

```
INFO - Model saved as <name_of_the_model>
```

Use this name to copy the file from container (you will also need to create `models` folder if it's not created yet):

```
docker cp <container_id>:/app/models/<name_of_the_model> ./models
```

### Inference process

Inference process can be done in two ways: on a local machine or using Docker.

#### Perform inference locally

Run this command to infer data using one of the models:

```
python3 inference/run.py <name_of_the_model>
```

After running the command, the results will be saved in the `results` folder on a .csv file under a name that will appear on one of log messages that looks like this:

```
INFO - Predictions are saved to <name_of_the_result_file>
```

#### Perform inference using Docker

Firstly, you have to build a Docker image, using this command:

```
docker build -f ./inference/Dockerfile --build-arg settings_name=settings.json -t inference_image .
```

When the image finishes building, you need to make a container by running this command:

```
docker run -it inference_image python3 inference/run.py <name_of_the_model>
```

Again, you need to find out the container id which you can do using Docker Desktop.

One of the log messages will show the name of the file:

```
INFO - Predictions are saved to <name_of_the_result_file>
```

Use this name to copy the file from container (you will also need to create `results` folder if it's not created yet):

```
docker cp <container_id>:/app/results/<name_of_the_result_file> ./results
```

## Troubleshooting

Some Linux users might come across a problem when running Dockerfile in training folder:

```
COPY failed: forbidden path outside the build context: ../data/
```

If this problem arises, you can resolve it by editing the dockerfile and subtituting `../data/` for `data/` (you might also need to do that with other paths that include double dots).