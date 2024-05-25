## Weed managers guide to Remote Detection
Understanding the opportunities and limitations of multi-resolution and multi-modal technologies for remote detection of weeds in heterogenous landscapes
![ai](https://github.com/Narmilan-A/Remote-Weed-detection/assets/140802455/fd66ff8f-abb0-4527-927e-81d46da6b8e6)

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure and Description](#repository-structure-and-description)
- [Installation](#installation)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [License](#license)
- [Contact](#contact)

## Introduction
This repository hosts the resources and findings from the project "Weed Managers Guide to Remote Detection: Understanding Opportunities and Limitations of Multi-Resolution and Multi-Modal Technologies for Remote Detection of Weeds in Heterogeneous Landscapes." Over the past three years, our collaborative research has focused on developing efficient, accessible, and cost-effective remote detection methods to improve weed management strategies. Our project employed advanced technologies, including high-resolution RGB, multispectral, and hyperspectral imaging, across various airborne platforms like drones and satellites. By leveraging artificial intelligence, including both classical machine learning and deep learning techniques, we successfully detected and mapped weed infestations in diverse conservation landscapes. Our study specifically targeted model weed systems such as hawkweed and bitou bush to assess the capabilities and limitations of these technologies. By sharing our research outcomes and practical insights, we aim to contribute to the development of improved and more feasible approaches for remote weed detection, ultimately supporting more effective weed management practices.

## Repository Structure and Description
The repository is organised into two main folders corresponding to the model weed detection studied: `Bitou bush` and `Hawkweed`. Each of these folders contains several sub-folders, each dedicated to different aspects of the project. Here's an overview of the structure and contents of each folder: 
#### Main Folders
- `Bitou bush`
- `Hawkweed`

Each of these main folders is accompanied by a `README.md` file that provides further details and instructions specific to the contents and usage of the files within. Each main folder includes the following sub-folders:

- **Ground truth labelling**: Contains the ground truth data and associated labeling files used for training and testing the detection models.

- **Multispectral_dataset**: Contains the multispectral imaging datasets utilised in the project.

- **Hyperspectral_dataset**: Holds the hyperspectral imaging datasets used in the project.

- **Model**: Includes the models used for training, validating, and testing weed detection algorithms.

- **Scripts**: Holds various scripts used throughout the project for data processing, model training, and evaluation.

- **Prediction**: Stores the prediction outputs from the trained models.

- **Report and Publication**: Includes reports, publications, and documentation generated from the project findings.

This structure ensures that all relevant resources, data, and documentation are organised systematically, making it easy for users to navigate and utilize the repository effectively.

## Installation
Before using the resources and scripts in this repository, ensure that you have the necessary dependencies installed. You can use the provided `.yml` file to set up a conda environment with the required packages. Please follow the steps below to create the environment:
1. Download the provided `environment.yml` file from this repository.
2. Open a terminal or command prompt.
3. Navigate to the directory where you downloaded the `environment.yml` file.
4. Run the following command to create a new conda environment using the specifications in the `environment.yml` file:
```shell
conda env create -f environment.yml
```
5. Once the environment is created, activate it using the following command:
```shell
conda activate <environment_name>
```
After completing these steps, you'll have the required dependencies installed in your conda environment, and you'll be ready to use the resources provided in this repository. If you encounter any issues during the setup process, feel free to reach out to us for assistance.

**For more information**: Refer to the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Datasets
This repository is a comprehensive resource for researchers and practitioners interested in leveraging UAV technology for precise weed detection. Featuring data on specific target weed species like Hawkweed and BitouBush, the repository delineates various site locations where detection efforts have been conducted. Utilising advanced UAV platforms such as the DJI M300 and DJI M600 and specialised sensor models such as the Micasense Altum (Multispectral camera) and Specim AFX VNIR (Hyperspectral camera), the repository showcases a variety of setups tailored for multispectral and hyperspectral analysis.

## Model details
In this project, a combination of classical machine learning (ML) techniques and deep learning (DL) architectures was employed for various computer vision tasks. Classical ML models such as Extreme Gradient Boosting (XGBoost), Random Forest(RF), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) were utilized alongside DL models like U-Net. These models were specifically applied for semantic segmentation tasks targeting different morphologies, including mixing of foliage and flowers, as well as individual identification of flowers and foliage. The diverse range of models utilised underscores the comprehensive approach taken to tackle weed detection challenges using both traditional and cutting-edge ML methodologies.

## License

## Contact
For inquiries, feedback, or collaboration opportunities, please feel free to reach out:

- **Email:** [narmilan.amarasingam@hdr.qut.edu.au](mailto:narmilan.amarasingam@hdr.qut.edu.au)
- **Issue Tracker:** [GitHub Issues](https://github.com/Narmilan-A)
- **LinkedIn:** [Narmilan Amarasingam](https://www.linkedin.com/in/narmilan-amarasingam-ab7086115/)
- **Project website:** [More information](https://www.csu.edu.au/research/gulbali/research/agricultural-innovation/projects/weed-managers-guide-to-remote-detection)

