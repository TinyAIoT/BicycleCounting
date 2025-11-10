# Model Training Sub-Repo

This repository contains all the Python code that is necessary for model training and fine-tuning

### Structure
```text
.
├── data                // image data and other data sources for model training
├── datasets            // downloaded or created datasets for model training (e.g., COCO)
├── configs             // run configurations in YAML format for training jobs
├── runs                // training runs' configurations including weights and evaluation plots
├── model_training      // main directory containing the code for model training
├── models              // directory for storing fine-tuned/trained models
└── notebooks           // notebooks directory for testing and evaluating stuff
```

### Setup

> [!IMPORTANT]  
> Run all commands from the `model-training` directory!

> [!NOTE]  
> Make sure to [install uv](https://docs.astral.sh/uv/getting-started/installation/) as a package and dependency manager.
> Run ``curl -LsSf https://astral.sh/uv/install.sh | sh`` from your terminal to install uv globally. Restart your terminal session after installation.

#### Dependencies

Run the ``setup.sh`` script for setting up the environment.
```bash
bash setup.sh
```

Activate the virtual environment by running:
```bash
source .venv/bin/activate
```

In order to test your setup, you can run a test training job by running
```bash
poe train configs/yolo11n_sample_config.yaml
```


### Start a new Training Job
All training jobs are started from a run configuration in YAML format. Every job gets started via the CLI with a dedicated shell command.

To start a new training job, you have to set up a new run configuration in the ``configs`` folder. 
Consider the ``configs/yolo11n_sample_config.yaml`` as a reference for your own training job configuration.
We make use of Ultralytics for running trainings of YOLO models. Thus, to see the Ultralytics website for a [full list of the training and augmentation arguments](https://docs.ultralytics.com/modes/train/#train-settings).

A new training job can then be started by running
```bash
poe train configs/PATH_TO_YOUR_YAML_CONFIG
```

> [!NOTE]  
> The ``poe`` command is part of the virtual environment of this sub-repository. Thus, make sure to activate the virtual environment and run this command from the ``model-training`` directory.


### CI Jobs
[poethepoet](https://poethepoet.natn.io/) is a CLI wrapper and allows to customize terminal pipelines. We make use of this package in order to configure CI tasks (e.g., linter, typing). GitHub Actions are configured for the same tasks.
Each job is configured in the [pyproject.toml](pyproject.toml) file.

To see a full list of CI tasks, run
```bash
poe
```

from the terminal.

Run all checks:
```bash
poe ci
```
