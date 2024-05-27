## Key Steps and Explanations: classical_ml_training_&_prediction.py

| Step | Description |
|------|-------------|
| 1.   | **Import Libraries** |
|      | Import necessary Python libraries such as NumPy, Matplotlib, GDAL, Scikit-learn, XGBoost, Joblib, Seaborn, and others for data processing, visualization, and machine learning tasks. |
| 2.   | **Define Vegetation Indices Function** |
|      | Define a function to calculate various vegetation indices such as NDVI, GNDVI, NDRE, GCI, MSAVI, and EXG using input satellite imagery. |
| 3.   | **Load and Preprocess Data** |
|      | Load input images and corresponding masks, preprocess the images by equalizing histograms, calculate vegetation indices using the defined function, and filter unlabelled data. |
| 4.   | **Resampling Data** |
|      | Resample the dataset to balance the classes by downsampling/up sampling the minority/majority classes. |
| 5.   | **Train-Test Split** |
|      | Split the dataset into training and testing sets using train_test_split function from scikit-learn. |
| 6.   | **Model Training** |
|      | Train machine learning models including Random Forest, Support Vector Machine (SVC), XGBoost, and K-Nearest Neighbors (KNN) using the training data. |
| 7.   | **Save Best Models** |
|      | Save the trained models with the best performance using joblib for future use. |
| 8.   | **Evaluate Models** |
|      | Evaluate the trained models using confusion matrices and classification reports, and save the results in a text file. Visualize confusion matrices as heatmaps. |
| 9.   | **Display and Export Predictions** |
|      | Display and export prediction results for each model, including input images, masks, and predicted images. Visualize predictions for multiple regions of interest (ROIs) using custom color maps. |

## Key Steps and Explanations: classical_ml_testing_&_prediction.py
| Step | Description |
|------|-------------|
| 1.   | **Import Libraries** Import necessary Python libraries including NumPy, Matplotlib, GDAL, Scikit-learn, Joblib, Seaborn, and others for data processing, visualization, and machine learning tasks. |
| 2.   | **Define Vegetation Indices Function** Define a function (`calculate_veg_indices`) to calculate various vegetation indices such as NDVI, GNDVI, NDRE, GCI, MSAVI, and EXG using input imagery. |
| 3.   | **Load and Preprocess Data** Load input images and corresponding masks, preprocess the images by equalizing histograms, calculate vegetation indices using the defined function, and filter unlabelled data. |
| 4.   | **Resampling Data** Separate the dataset into classes, resample to balance the classes by downsampling/upsampling the minority/majority classes, and combine them into a single dataset. |
| 5.   | **Predict Using Best Models** Load the saved best models (RF, SVM, XGB, KNN) and make predictions on the training data. |
| 6.   | **Evaluate Models** Calculate confusion matrices and classification reports for each model, visualize confusion matrices as heatmaps, and save the results in a text file. |
| 7.   | **Display and Export Predictions** Display and export prediction results including input images, masks, and predicted images for each model. Visualize predictions for multiple regions of interest (ROIs) using custom color maps. |
| 8.   | **Export Prediction Results for Selected Model** After selecting the best model (XGB), export the prediction results for all ROIs using custom color maps. |
