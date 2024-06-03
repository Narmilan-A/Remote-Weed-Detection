## Processing pipeline for DL U-Net model training
![image](https://github.com/Narmilan-A/Remote-Weed-detection/assets/140802455/0c1300b6-cb96-423b-948b-c8b7e145a8c1)

### Brief explanation of unet_training.py
| Main steps                           | Description                                                                                                                  |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Import Libraries                     | Import Python libraries: NumPy, OS, Pandas, OpenCV, Matplotlib, Seaborn, scikit-image for image processing, GDAL for geospatial data handling, scikit-learn and Keras for model development. |
| Calculate Vegetation Indices         | Define `calculate_veg_indices` function to compute various vegetation indices from multispectral images.                          |
| Tile Images and Masks and define paths | Split images and masks into tiles with specified size and overlap, and specify root directories for images and masks. Retrieve files and preprocess tiles. |
| Convert Masks to Categorical         | Convert mask data to categorical representation using one-hot encoding.                                                             |
| Train-Test Split                     | Split data into training and testing sets using scikit-learn’s `train_test_split` function.                                        |
| U-Net Architecture                   | Define U-Net model architecture with convolutional and deconvolutional layers for semantic segmentation.                           |
| Model Compilation                    | Compile U-Net model with Adam optimizer and categorical cross-entropy loss.                                                         |
| Model Training                       | Train model on training data and validate it using testing data. Save best model checkpoint based on validation loss.               |
| Confusion Matrix and Classification Report | Predict on test data, calculate confusion matrix and classification report. Plot confusion matrix as heatmap.                 |
| IoU Calculation                      | Calculate and save Intersection over Union (IoU) for each class in segmentation.                                                    |

### Brief explanation of unet_testing.py
| Main steps                           | Description                                                                                                                  |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Import Necessary Libraries          | Import Python libraries: NumPy, OS, Pandas, OpenCV, Matplotlib, Seaborn, scikit-image, GDAL, scikit-learn, and Keras for various tasks. |
| Define the Root Directory           | Define `root folder` variable as the absolute path where the code operates.                                                         |
| Load U-Net Model                    | Load U-Net model from saved file using Keras `load_model` function.                                                                 |
| Calculate Vegetation Indices        | Define `calculate_veg_indices` function to compute vegetation indices for input images.                                             |
| Tile Images and Masks               | Set tile size and overlap percentage, specify root directory for input images and masks, retrieve files, split into tiles, preprocess, and append to lists. |
| Predict and Evaluate Using Confusion Matrix | Use loaded U-Net model to predict masks, compute and visualize confusion matrix, save heatmap.                                |
| Confusion Matrix and Classification Report | Predict on test data, compute confusion matrix and classification report, save both to files.                                  |
| Calculate and Save IoU for Each Class | Calculate IoU for each class, save results to text file, compute and save average IoU.                                             |

### Brief explanation of unet_prediction.py
| Main steps                           |                  Description                                                                                                 |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Import Necessary Libraries          | Import Python libraries: NumPy, OS, Pandas, OpenCV, Matplotlib, Seaborn, scikit-image, GDAL, scikit-learn, and Keras for various tasks. |
| Define the Root Directory           | Define `root folder` variable as the absolute path where the code operates.                                                         |
| Load U-Net Model                    | Load U-Net model from saved file using Keras `load_model` function.                                                                 |
| Calculate Vegetation Indices        | Define `calculate_veg_indices` function to compute vegetation indices for input images.                                             |
| Tile Images and Masks               | Set tile size and overlap percentage, specify root directory for input images and masks, retrieve files, split into tiles, preprocess, and append to lists. |
| Prediction                          | Loop through each image file, apply preprocessing steps, extract patches, resize patches if necessary, predict using U-Net model, merge patches, save predicted image in ENVI format. |

### Brief explanation of label_rasterizing.py
| Main steps                  | Description                                                                                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------|
| Import Libraries           | Import necessary libraries: `gdal`, `ogr` from the `osgeo` package, `os`, and `glob` for file and directory operations. |
| Shape to raster function   | This function converts a shapefile to a raster mask based on a reference image's spatial extent and resolution. |
| Main function              | The main function orchestrates the rasterization process by iterating through input images and calling the shape to raster function for each, saving the output masks. |

### Brief explanation of unet_kfold_cross_validation.py
| Main steps                  | Description                                                                                                   |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------|
| Import Libraries           | Import necessary libraries: `numpy`, `os`, `pandas`, `cv2`, `matplotlib`, `seaborn`, `exposure` from skimage, `convolve` from `scipy.ndimage`, `time`, `random`, and `tensorflow` modules. Also, import specific functions from `osgeo` (`gdal`) and scikit-learn (`train_test_split`, `KFold`). Import necessary functions and classes from Keras. |
| Define Parameters          | Set various parameters controlling the operations, such as applying vegetation indices, Gaussian blur, mean filter, convolution, and band deletion. Specify image dimensions, tile size, overlap percentage, test size, learning rate, batch size, and epochs. Define class names and number of classes.                                           |
| Define Functions           | Define functions for post-index calculation, calculation of vegetation indices, applying Gaussian blur, applying mean filter, and defining the UNet model architecture.                                                                                                                                                                                       |
| Load and Preprocess Data   | Load image and mask files, filter them based on dimensions, apply preprocessing steps such as equalization, vegetation index calculation, band deletion, convolution, Gaussian blur, and mean filter. Split data into training and testing sets.                                                                   |
| Train Model                | Define and compile the UNet model, apply K-Fold cross-validation, train the model for each fold, save the best model based on validation loss, and evaluate each fold's performance. Save model outcomes and evaluation results to files.                                                                  |
| Visualize Results          | Plot and save graphs of validation accuracy and loss across folds, including average and standard deviation lines.                                                                                                                             |
## NOTE
### Improving Model Performance

To enhance the performance of the model based on the provided code, the following strategies can be considered:

1. **Data Augmentation**: Augmenting the dataset by applying transformations such as rotation, scaling, flipping, and cropping can help diversify the training data, thereby improving the model's ability to generalize to unseen data.

2. **Hyperparameter Tuning**: Experimenting with different learning rates, batch sizes, and epochs can optimize the model's training process. Techniques like learning rate scheduling and early stopping can also be employed to fine-tune training dynamics.

3. **Architecture Modifications**: Adjusting the architecture of the neural network, such as adding more layers, increasing the number of filters, or using deeper network architectures, can enhance the model's capacity to capture intricate patterns in the data.

4. **Regularization Techniques**: Incorporating regularization techniques like dropout or weight decay can prevent overfitting by imposing constraints on the model parameters during training.

5. **Transfer Learning**: Leveraging pre-trained models and fine-tuning them on the specific task at hand can expedite the training process and potentially improve performance, especially when working with limited training data.

6. **Ensemble Learning**: Combining predictions from multiple models, either through averaging or stacking, can often lead to superior performance compared to individual models, by leveraging diverse sources of information.

7. **Optimized Data Preprocessing**: Fine-tuning data preprocessing steps such as normalization, feature scaling, and handling missing values can ensure that the model receives high-quality input data, leading to improved performance.

8. **Vegetation Indices**: Incorporating vegetation indices such as NDVI (Normalized Difference Vegetation Index), NDRE (Normalized Difference Red Edge), and MSAVI (Modified Soil-Adjusted Vegetation Index) can provide additional spectral information to the model, aiding in the discrimination of different vegetation types and enhancing classification accuracy.

9. **Kernel and Filter Operations**: Applying kernel operations such as low-pass averaging and Gaussian blur can help smooth images, reduce noise, and enhance feature extraction capabilities, thereby improving the model's ability to discern meaningful patterns in the data.

10. **Control Parameters**: Fine-tuning control parameters such as applying vegetation indices, Gaussian blur, mean filter, convolution, band deletion, and filtering based on image dimensions can optimize the preprocessing pipeline and improve model performance.

By systematically exploring and implementing these strategies, the model's performance can be significantly enhanced, leading to better accuracy and robustness in weed detection tasks.

### Mitigating Memory Errors in Training and Prediction
Memory errors during both training and prediction can hinder the performance and efficiency of machine learning models. This README provides recommendations for mitigating memory errors in training and prediction processes.
#### Training
To address memory errors during training, consider the following:
- **Reduce Number of Channels**: Eliminate some Variable Importance (VI) channels to reduce memory usage.
- **Decrease Patch Size**: Reduce patch size from 256 to 128 or smaller, while also decreasing the overlap.
- **Trim Down Model Layers**: Reduce the number of layers in the model architecture, transitioning from higher numbers (e.g., 1024) to lower ones (e.g., 64, 256, or 512).
- **Adjust Batch Size and Learning Rate**: Modify batch size and learning rate to optimize memory usage during training.
- **Utilize High-Performance Computing (HPC)**: Consider using powerful GPU setups like A100, QUT HPC, or Google Colab Pro Plus for successful training with large multispectral data and deep learning models.

#### Prediction
For memory-efficient prediction, follow these recommendations:
- **Employ Training Techniques**: Apply the same training techniques mentioned above during prediction to optimize memory usage.
- **Reduce Number of Regions of Interest (ROIs)**: Use a smaller number of ROIs during prediction to reduce memory overhead.
- **Work with Reduced Data Area**: Utilize smaller subsets of data or reduced areas of orthomosaic data for prediction tasks.
- **Merge Data Effectively**: Merge smaller subsets of data using tools like Arc GIS Pro to handle memory constraints.
