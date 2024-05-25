### Brief explanation of hsi_classical_ml_training.py
| Main Steps                | Brief Explanation                                                                                                                                                                      |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Import Libraries          | Import necessary libraries: `numpy`, `matplotlib.pyplot`, `gdal` from `osgeo`, `train_test_split`, `RandomForestClassifier`, `confusion_matrix`, `classification_report`, `seaborn`, `SVC`, `xgb`, `KNeighborsClassifier`, `joblib`, `ListedColormap`, `exposure`, `resample`, `imsave`, `PCA`, `StandardScaler`, `convolve`. |
| Read Bands File           | Read the selected bands information from a text file and convert it into a list.                                                                                                       |
| Load Images and Masks     | Load ROI images and corresponding masks, apply preprocessing such as equalization and low-pass filter using convolution, and store them in lists.                                    |
| Preprocess Data           | Filter unlabelled data, extract features from images, and concatenate them into training data.                                                                                         |
| Resample Dataset          | Upsample minority classes to balance the dataset.                                                                                                                                      |
| Train-Test Split          | Split the data into training and testing sets.                                                                                                                                          |
| Data Normalization        | Scale the features for Euclidean distance calculation.                                                                                                                                  |
| Train Classifiers         | Define and fit Random Forest, XGBoost, KNN, and SVM classifiers to the training data.                                                                                                  |
| Evaluate Models           | Calculate confusion matrices and classification reports for each model using the test data. Save outcomes to files and visualize confusion matrices.                                   |

### Brief explanation of hsi_classical_ml_testing.py
| Main Steps              | Brief Explanation                                                                                                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Import Libraries        | Import necessary libraries: `numpy`, `matplotlib.pyplot`, `gdal`, `confusion_matrix`, `classification_report`, `seaborn`, `joblib`, `ListedColormap`, `exposure`, `resample`, `imsave`, `PCA`, `convolve`.                           |
| Read Bands File         | Read selected bands information from a text file and convert it into a list.                                                                                                       |
| Load Images and Masks   | Load ROI images and corresponding masks, apply preprocessing such as equalization and low-pass filter using convolution, and store them in lists.                              |
| Preprocess Data         | Filter unlabelled data, extract features from images, and concatenate them into training data.                                                                                     |
| Resample Dataset        | Upsample minority classes to balance the dataset.                                                                                                                                  |
| Load Saved Models       | Load trained models saved during training phase.                                                                                                                                   |
| Validate Models         | Predict labels for training data using loaded models, calculate confusion matrices and classification reports for each model, and save outcomes to files.                         |

### Brief explanation of hsi_classical_ml_prediction.py
| Step                                      | Brief Explanation                                                                                                                          |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Importing Libraries                      | Importing necessary Python libraries including NumPy, Matplotlib, GDAL, joblib, and others.                                        |
| Loading Saved Models                     | Loading pre-trained machine learning models (KNN, RF, SVC, XGB) from the specified file paths.                                      |
| Displaying and Exporting Prediction Results | Displaying and exporting prediction results using custom colormaps for masks and predictions.                                       |
| Selecting Bands                          | Reading selected bands information from a file and processing the input images and masks accordingly.                               |
| Applying Low-pass Filter                 | Applying a 7x7 low-pass filter to input images using convolution and preprocessing them for prediction.                             |
| Plotting Input, Mask, and Prediction    | Plotting input images, masks, and predictions for each ROI using different machine learning models (RF, SVC, XGB, KNN).             |
| Saving Prediction Images                 | Saving the predicted images in ENVI format and exporting them as PNG images for visualization and further analysis.                  |
| Creating Custom Legend                   | Creating a custom legend for the predictions with color-coded labels for better interpretation.                                    |
| Displaying and Exporting Results         | Displaying the final results, including input images, masks, and predictions, and exporting them as PNG images for further analysis. |

### Brief explanation of hsi_band-selection.py
| Step                                      | Description                                                                                                                          |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Importing Libraries                      | Importing necessary Python libraries including NumPy, Matplotlib, and GDAL.                                                          |
| Loading Image and Mask                   | Opening the image and mask files using GDAL and converting them to NumPy arrays.                                                     |
| Computing Spectral Values                | Computing mean spectral values for each class by iterating through each band.                                                        |
| Calculating Differences between Classes  | Calculating the absolute differences between class curves for each band.                                                             |
| Plotting Spectral Signature Curves      | Plotting spectral signature curves for each class with respect to band number.                                                       |
| Selecting Top Bands                      | Selecting top bands with the most significant spectral differences between classes and saving the results to a file.                  |
| Plotting Spectral Signature Curves with Annotations | Plotting spectral signature curves for each class with annotations of selected bands showing the highest differences.             |
