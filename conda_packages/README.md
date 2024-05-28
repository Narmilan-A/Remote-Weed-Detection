## Machine Learning Environment Configurations

This repository contains the configurations for running Classical Machine Learning (ML) and Deep Learning (DL) models on different hardware setups. Below are the details of the hardware environments used for running the respective models.

### Classical ML (XGB / RF / SVM / KNN)

#### High performance computing (HPC) (Optinal)
- **RAM**: 128GB
- **CPU Cores (ncpus)**: 16

#### Local computer
- **RAM**: 16GB
- **Processor**: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, 1.80 GHz

### Deep Learning (UNet)

#### HPC
- **GPU**: A100 NVIDIA
- **GPU Memory**: 40GB
- **Total GPUs**: 48
- **GPUs per node**: 8
- **RAM**: 32GB
- **CPU Cores (ncpus)**: 4

### Conda Environment Packages

The `conda_environment_package` folder contains two `.yml` files for setting up the necessary conda environments:

1. **`environment_local.yml`**: This file is used to set up the conda environment for running scripts on a local computer.
2. **`environment_hpc.yml`**: This file is used to set up the conda environment for running scripts on the HPC cluster (platform: linux-64).

This Conda environment file specifies numerous packages that are commonly used in various machine learning frameworks. Below is a summary of how different deep learning (DL) training frameworks and libraries are represented:
- **TensorFlow**: TensorFlow is a popular open-source library for machine learning and deep learning. It provides comprehensive tools and libraries for building and training ML models.
- **Keras**: Keras simplifies the creation, training, and deployment of neural networks by providing a user-friendly interface and seamless integration with various deep learning frameworks like TensorFlow, Theano, and CNTK.
- **Data Handling and Visualization**: Pandas,Matplotlib, Seaborn, Scikit-image, OpenCV
- **Development and Testing**: Jupyter Notebook/Lab, IPython
- **Supporting Libraries**: Numpy, Scipy, Joblib
- **Geospatial Image Processing Tools**: GDAL
- **classification tasks**: XGBoost, Scikit-learn

You can get started quickly by following one of the provided methods to install the required packages after creating a Conda environment.

#### Method 1: Using Conda (recommended)

1. Create a Conda environment using the provided environment.yml file for your specific environment:
    - For local computing:
        ```
        conda env create -f cond_packages/local_environment.yml
        ```
    - For HPC:
        ```
        conda env create -f cond_packages/hpc_environment.yml
        ```

2. Activate the newly created environment:
    ```
    conda activate my_environment
    ```

#### Method 2:  Manual Installation

1. Activate your Conda environment:
    ```
    conda activate my_environment
    ```

2. Install each required package individually using Conda or pip:
    ```
    conda install package_name
    ```
    or
    ```
    pip install package_name
    ```

3. Repeat step 2 for each package listed in the project's requirements.

4. Verify that all packages are installed correctly by running:
    ```
    conda list
    ```
    or
    ```
    pip list
    ```

### Additional Information

- If you encounter any issues during installation or while running the project, please refer to the documentation or the project's issue tracker for assistance.
- For more detailed instructions or troubleshooting, please visit the project's GitHub repository or contact the maintainers directly.

### Useful Links

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Scikit-image Documentation](https://scikit-image.org/docs/stable/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [IPython Documentation](https://ipython.org/documentation.html)
- [Numpy Documentation](https://numpy.org/doc/)
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/reference/)
- [Joblib Documentation](https://joblib.readthedocs.io/en/latest/)
- [GDAL Documentation](https://gdal.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

These links provide access to the official documentation or website of each tool or library, where users can find detailed information, tutorials, and examples for usage.
