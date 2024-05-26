# Machine Learning and Deep Learning Environment Configurations

This repository contains the configurations for running Classical Machine Learning (ML) and Deep Learning (DL) models on different hardware setups. Below are the details of the hardware environments used for running the respective models.

## Classical ML (XGB / RF / SVM / KNN)

### High performance computing (HPC) (Optinal)
- **RAM**: 128GB
- **CPU Cores (ncpus)**: 16

### Local computer
- **RAM**: 16GB
- **Processor**: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, 1.80 GHz

## Deep Learning (UNet)

### HPC
- **GPU**: A100 NVIDIA
- **GPU Memory**: 40GB
- **Total GPUs**: 48
- **GPUs per node**: 8
- **RAM**: 32GB
- **CPU Cores (ncpus)**: 4

## Conda Environment Packages

The `conda_environment_package` folder contains two `.yml` files for setting up the necessary conda environments:

1. **`environment_local.yml`**: This file is used to set up the conda environment for running scripts on a local computer.
2. **`environment_hpc.yml`**: This file is used to set up the conda environment for running scripts on the HPC cluster.
