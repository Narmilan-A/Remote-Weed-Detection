## Processing pipeline for DL U-Net model training
![image](https://github.com/Narmilan-A/Remote-Weed-detection/assets/140802455/0c1300b6-cb96-423b-948b-c8b7e145a8c1)

## Brief explanation of unet_training.py
| Main steps                           | Brief Explanation                                                                                                                  |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Import Libraries                     | - General Python libraries: NumPy, OS, Pandas, OpenCV, Matplotlib, Seaborn, scikit-image.                                         |
|                                      | - Geospatial data handling: GDAL.                                                                                                  |
|                                      | - Model development: scikit-learn, Keras.                                                                                          |
| Calculate Vegetation Indices         | Define `calculate_veg_indices` function to compute vegetation indices from multispectral images.                                    |
| Tile Images and Masks and define paths | - Split images and masks into tiles with specified size and overlap.                                                               |
|                                      | - Specify root directory for images and masks. Retrieve files. Preprocess tiles.                                                   |
| Convert Masks to Categorical         | Convert mask data to categorical representation using one-hot encoding.                                                             |
| Train-Test Split                     | Split data into training and testing sets.                                                                                         |
| U-Net Architecture                   | Define U-Net model architecture for semantic segmentation.                                                                         |
| Model Compilation                    | Compile U-Net model with Adam optimizer and categorical cross-entropy loss.                                                         |
| Model Training                       | Train model using training data. Validate with testing data. Save best model checkpoint based on validation loss.                   |
| Confusion Matrix and Classification Report | Predict on test data. Calculate confusion matrix and classification report. Plot and save confusion matrix as heatmap.         |
| IoU Calculation                      | Calculate and save Intersection over Union (IoU) for each class in segmentation.                                                    |
