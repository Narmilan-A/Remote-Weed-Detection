#----------------------------------------------------------------------#
# Import general python libraries
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib
from matplotlib.colors import ListedColormap
from skimage import exposure 
from sklearn.utils import resample

#----------------------------------------------------------------------#
# Define function to calculate vegetation indices
def calculate_veg_indices(input_img):
    RedEdge = input_img[:, :, 3]
    nir = input_img[:, :, 4]
    red = input_img[:, :, 2]
    green = input_img[:, :, 1]
    blue = input_img[:, :, 0]

    ndvi = (nir - red) / (nir + red)
    gndvi = (nir - green) / (nir + green)
    ndre = (nir - RedEdge) / (nir + RedEdge)
    gci = (nir)/(green) - 1
    msavi = ((2 * nir) + 1 -(np.sqrt(np.power((2 * nir + 1), 2) - 8*(nir - red))))/2
    exg = ((2*green)-red-blue)/(red+green+blue)

    veg_indices = np.stack((ndvi,), axis=2)
    return veg_indices
#----------------------------------------------------------------------#
input_imgs = []
input_masks = []
rois = [5]

for i in rois:
    input_img_file = 'C:/Users/N10837647/OneDrive - Queensland University of Technology/Desktop/nz_hw_rs/flower_classification/rois/nz_hw_roi_{}.tif'.format(i)
    input_mask_file = 'C:/Users/N10837647/OneDrive - Queensland University of Technology/Desktop/nz_hw_rs/flower_classification/masks/nz_hw_mask_{}.tif'.format(i)
    ds_img = gdal.Open(input_img_file)
    ds_mask = gdal.Open(input_mask_file)
    input_img = np.array([ds_img.GetRasterBand(j+1).ReadAsArray() for j in range(5)])
    input_img= np.transpose(input_img, (1,2,0))
    input_img = exposure.equalize_hist(input_img)
    veg_indices = calculate_veg_indices(input_img)
    input_img = np.concatenate((input_img, veg_indices), axis=2)
    input_mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(int)
    input_imgs.append(input_img)
    input_masks.append(input_mask)
#----------------------------------------------------------------------#
# Preprocess the data
X = []
y = []
for i in range(len(rois)):
    # Filtering unlabelled data
    gt_mask = ((input_masks[i] > 0))
    # Filter unlabelled data from the source image and store their values in the 'X' features variable
    x_array = input_imgs[i][gt_mask, :]
    # Select only labelled data from the labelled image and store their values in the 'y' labels variable
    y_array = input_masks[i][gt_mask]
    # Covert to array format
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    X.append(x_array)
    y.append(y_array)

# Concatenate the arrays
X_train = np.concatenate(X)
y_train = np.concatenate(y)

print('"X" matrix size: {sz}'.format(sz=X_train.shape))
print('"y" array  size: {sz}'.format(sz=y_train.shape))
#----------------------------------------------------------------------#
# Resampling of dataset

# Separate the three classes
id_0 = np.where(y_train == 0)[0]
id_1 = np.where(y_train == 1)[0]
id_2 = np.where(y_train == 2)[0]
id_3 = np.where(y_train == 3)[0]

# Print the pixels numbers
print("Number of pixels in id_1: :"f"{len(id_0)}")
print("Number of pixels in id_1: :"f"{len(id_1)}")
print("Number of pixels in id_2: :"f"{len(id_2)}")
print("Number of pixels in id_3: :"f"{len(id_3)}")

# Resample : Downsample / upsample the class

id_2_downsampled = resample(id_2, replace=True, n_samples=len(id_1), random_state=42)


# Print the pixels numbers after resampling
print("Number of pixels in id_1 after resampling: :"f"{len(id_1)}")
print("Number of pixels in id_2 after resampling: :"f"{len(id_2_downsampled)}")
print("Number of pixels in id_3 after resampling: :"f"{len(id_3)}")

# Combine the three classes into a single dataset
X = np.concatenate((X_train[id_1], X_train[id_2_downsampled], X_train[id_3]), axis=0)
y = np.concatenate((y_train[id_1], y_train[id_2_downsampled], y_train[id_3]), axis=0)

# Shuffle the combined dataset
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

X_train=X
y_train=y
#----------------------------------------------------------------------#
# Load the saved model
best_knn_model = joblib.load('best_knn_model.pkl')
best_rf_model = joblib.load('best_rf_model.pkl')
best_svm_model = joblib.load('best_svm_model.pkl')
best_xgb_model = joblib.load('best_xgb_model.pkl')
#----------------------------------------------------------------------#
# Make predictions on the test data using the best models
print('Predicting test data using the best models... ', end="", flush=True)

knn_preds = best_knn_model.predict(X_train)
rf_preds = best_rf_model.predict(X_train)
svm_preds = best_svm_model.predict(X_train)
xgb_preds = best_xgb_model.predict(X_train)

print("Predicted X_train")
#------------------------------------------------------------------#
# # # List of models and their corresponding classifiers
# # print('Estmatng the model performances... ', end="", flush=True)

models = ['RF', 'SVM', 'XGB', 'KNN']
classifiers = [best_rf_model, best_svm_model, best_xgb_model, best_knn_model]
labels = ['Hw Flowers', 'Background']

# # Loop through each model
# # for i, model in enumerate(models):
# #     print(f'Calculating metrics for {model} ...')
    
# #     Predicting X_test using the current model
# #     y_pred = classifiers[i].predict(X_train)
    
# #     Calculate the confusion matrix for the current model
# #     cm = confusion_matrix(y_train, y_pred)
# #     print(f'Confusion matrix of {model}:')
# #     print(cm)
    
# #     Plot the confusion matrix for the current model using heatmap()
# #     plt.figure()
# #     sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
# #     plt.title(f'Confusion matrix heatmap of {model}')
# #     plt.xlabel('Predicted labels')
# #     plt.ylabel('True labels')
# #     plt.savefig(f'validation model outcomes/CM_heatmap_{model}.png', bbox_inches='tight')
# #     plt.show()
    
# #     Develop classification report only using class id=1,2,3
# #     cr = classification_report(y_train, y_pred, target_names=['Target_Vegetation', 'Other_Vegetation', 'Non_vegetation'])
# #     print(f'Classification report of {model}:')
# #     print(cr)
# # print("Estimated")
# # ------------------------------------------------------------------#
# # create a folder to save the output files
# # import os
# # if not os.path.exists('validation model outcomes'):
# #     os.makedirs('validation model outcomes')

# # create an empty string to store all the classification reports
# # all_cr = ''

# # create an empty list to store all the confusion matrices
# # all_cm = []

# # Loop through each model
# # for i, model in enumerate(models):
# #     print(f'Calculating metrics for {model} ...')
    
# #     Predicting X_test using the current model
# #     y_pred = classifiers[i].predict(X_train)
    
# #     Calculate the confusion matrix for the current model
# #     cm = confusion_matrix(y_train, y_pred)
# #     print(f'Confusion matrix of {model}:')
# #     print(cm)
    
# #     Save the confusion matrix to the all_cm list
# #     all_cm.append(cm)
    
# #     Plot the confusion matrix for the current model using heatmap()
# #     plt.figure()
# #     sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
# #     plt.title(f'Confusion matrix heatmap of {model}')
# #     plt.xlabel('Predicted labels')
# #     plt.ylabel('True labels')
# #     plt.savefig(f'validation model outcomes/CM_heatmap_{model}.png', bbox_inches='tight')
# #     plt.show()
    
# #     Develop classification report only using class id=1,2,3
# #     cr = classification_report(y_train, y_pred, target_names=['Target_Vegetation', 'Other_Vegetation', 'Non_vegetation'])
# #     print(f'Classification report of {model}:')
# #     print(cr)
    
# #     Append the classification report to the all_cr string
# #     all_cr += f'Classification report of {model}:\n{cr}\n\n'

# # Save all the classification reports and confusion matrices in one text file
# # with open('validation model outcomes/All_CR_CM.txt', 'w') as f:
    
# #     Write the classification reports to the file
# #     f.write('--- Classification Reports ---\n\n')
# #     f.write(all_cr)
    
# #     Write the confusion matrices to the file
# #     f.write('\n\n--- Confusion Matrices ---\n\n')
# #     for i, model in enumerate(models):
# #         f.write(f'Confusion matrix of {model}:\n{all_cm[i]}\n\n')

# # print("Estimated")
#------------------------------------------------------------------#
# List of models and their corresponding classifiers
print('Evaluating models... ', end="", flush=True)

models = ['RF', 'SVM', 'XGB', 'KNN']
classifiers = [best_rf_model, best_svm_model, best_xgb_model, best_knn_model]
#classifiers = [best_rf_model, best_xgb_model, best_svm_model, best_knn_model]
labels = ['Hw Flowers', 'Background']

# create a folder to save the output files
import os
if not os.path.exists('validation model outcomes'):
    os.makedirs('validation model outcomes')

# create an empty string to store all the classification reports
all_cr = ''

# create an empty list to store all the confusion matrices
all_cm = []

# Create a 2x2 figure for the heatmaps
fig, axs = plt.subplots(2, 2, figsize=(10,8))

# Loop through each model
for i, model in enumerate(models):
    print(f'Calculating metrics for {model} ...')
    
    # Predicting X_test using the current model
    y_pred = classifiers[i].predict(X_train)
    
    # Calculate the confusion matrix for the current model
    cm = confusion_matrix(y_train, y_pred)
    print(f'Confusion matrix of {model}:')
    print(cm)
    
    # Save the confusion matrix to the all_cm list
    all_cm.append(cm)

    # Set the font size for the annotations
    annot_font_size = 17

   # Plot the confusion matrix for the current model using heatmap()
    ax = axs[i//2, i%2] # Select the corresponding axis for the current model
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels, ax=ax, linewidths=.5, linecolor='black', annot_kws={'fontsize': annot_font_size})
    ax.set_title(f'Confusion matrix heatmap of {model}', fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    ax.set_xticks(np.arange(2) + 0.5)
    ax.set_xticklabels(labels, fontsize=15)

    ax.set_yticks(np.arange(2) + 0.5)
    ax.set_yticklabels(labels, fontsize=15)
 
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)

    ax.set_frame_on(True)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.set_frame_on(True)
    
    # Develop classification report only using class id=1,2,3
    cr = classification_report(y_train, y_pred, target_names=['Hw Flowers', 'Background'])
    print(f'Classification report of {model}:')
    print(cr)
    
    # Append the classification report to the all_cr string
    all_cr += f'Classification report of {model}:\n{cr}\n\n'

# Save all the classification reports and confusion matrices in one text file
with open('validation model outcomes/All_CR_CM_validation.txt', 'w') as f:
    
    # Write the classification reports to the file
    f.write('--- Classification Reports ---\n\n')
    f.write(all_cr)
    
    # Write the confusion matrices to the file
    f.write('\n\n--- Confusion Matrices ---\n\n')
    for i, model in enumerate(models):
        f.write(f'Confusion matrix of {model}:\n{all_cm[i]}\n\n')

# Save the figure
plt.tight_layout()
plt.savefig('validation model outcomes/CM_heatmap_subplot_validation.png', bbox_inches='tight', dpi=300)
plt.show()

print("Estimated")
#------------------------------------------------------------------#
# Display and export the prediction results

print('Displaying and exporting the prediction results... ', end="", flush=True)

# Define the custom colormaps for masks
colors_mask = ['#000000', '#ff0000', '#00ff00', '#0000ff']
cmap_mask = ListedColormap(colors_mask)

# Define the custom colormaps for predictions
colors_pred = ['#ff0000', '#00ff00', '#0000ff']
cmap_pred = ListedColormap(colors_pred)

# Plotting the input image, mask and prediction for all the models
rois = [1,2,5,7,8,9]

fig, axs = plt.subplots(len(rois), 6, figsize=(21, 20))

for i, roi in enumerate(rois):
    input_img_file = 'C:/Users/N10837647/OneDrive - Queensland University of Technology/Desktop/nz_hw_rs/Foliage_classification/rois/nz_hw_roi__{}.tif'.format(roi)
    input_mask_file = 'C:/Users/N10837647/OneDrive - Queensland University of Technology/Desktop/nz_hw_rs/Foliage_classification/masks/nz_hw_mask_{}.tif'.format(roi)
    ds_img = gdal.Open(input_img_file)
    ds_mask = gdal.Open(input_mask_file)
    input_img = np.array([ds_img.GetRasterBand(j+1).ReadAsArray() for j in range(5)])
    input_img = np.transpose(input_img, (1,2,0))
    input_img = exposure.equalize_hist(input_img)
    input_mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(int)
    veg_indices = calculate_veg_indices(input_img)
    input_img = np.concatenate((input_img, veg_indices), axis=2)
    input_mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(int)
    input_imgs.append(input_img)
    input_masks.append(input_mask)

    input_prediction_2d_RF = None
    input_prediction_2d_SVC = None
    input_prediction_2d_XGB = None
    input_prediction_2d_KNN = None

    # Predict using the classifier models
    input_img_hist_array = input_img[np.newaxis, ...]
    input_img_hist_array_2d = input_img_hist_array.reshape(-1, input_img_hist_array.shape[-1])

    if 'best_rf_model' in locals():
        input_prediction_RF = best_rf_model.predict(input_img_hist_array_2d)
        input_prediction_2d_RF = input_prediction_RF.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_RF = 'validation model outcomes/pred_geoinfo/RF_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_RF = driver.Create(pred_image_file_RF, input_prediction_2d_RF.shape[1], input_prediction_2d_RF.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_RF.GetRasterBand(1).WriteArray(input_prediction_2d_RF)
        pred_image_ds_RF.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_RF.SetProjection(ds_img.GetProjection())
        pred_image_ds_RF = None

    if 'best_svm_model' in locals():
        input_prediction_SVC = best_svm_model.predict(input_img_hist_array_2d)
        input_prediction_2d_SVC = input_prediction_SVC.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_SVC = 'validation model outcomes/pred_geoinfo/SVC_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_SVC = driver.Create(pred_image_file_SVC, input_prediction_2d_SVC.shape[1], input_prediction_2d_SVC.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_SVC.GetRasterBand(1).WriteArray(input_prediction_2d_SVC)
        pred_image_ds_SVC.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_SVC.SetProjection(ds_img.GetProjection())
        pred_image_ds_SVC = None

    if 'best_xgb_model' in locals():
        input_prediction_XGB = best_xgb_model.predict(input_img_hist_array_2d)
        input_prediction_2d_XGB = input_prediction_XGB.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_XGB = 'validation model outcomes/pred_geoinfo/XGB_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_XGB = driver.Create(pred_image_file_XGB, input_prediction_2d_XGB.shape[1], input_prediction_2d_XGB.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_XGB.GetRasterBand(1).WriteArray(input_prediction_2d_XGB)
        pred_image_ds_XGB.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_XGB.SetProjection(ds_img.GetProjection())
        pred_image_ds_XGB = None

    if 'best_knn_model' in locals():
        input_prediction_KNN = best_knn_model.predict(input_img_hist_array_2d)
        input_prediction_2d_KNN = input_prediction_KNN.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_KNN = 'validation model outcomes/pred_geoinfo/KNN_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_KNN = driver.Create(pred_image_file_KNN, input_prediction_2d_KNN.shape[1], input_prediction_2d_KNN.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_KNN.GetRasterBand(1).WriteArray(input_prediction_2d_KNN)
        pred_image_ds_KNN.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_KNN.SetProjection(ds_img.GetProjection())
        pred_image_ds_KNN = None

    # Display the input, mask, and predicted images for the current ROI
    axs[i, 0].imshow(input_img[:, :, [2, 1, 0]])
    axs[i, 0].set_title('Input Image (ROI {})'.format(roi))
    axs[i, 1].imshow(input_mask, cmap=cmap_mask)
    axs[i, 1].set_title('Mask Image (ROI {})'.format(roi))
    if 'best_rf_model' in locals():
        axs[i, 2].imshow(input_prediction_2d_RF, cmap=cmap_pred)
        axs[i, 2].set_title('RF Prediction (ROI {})'.format(roi))
    if 'best_svm_model' in locals():
        axs[i, 3].imshow(input_prediction_2d_SVC, cmap=cmap_pred)
        axs[i, 3].set_title('SVC Prediction (ROI {})'.format(roi))
    if 'best_xgb_model' in locals():
        axs[i, 4].imshow(input_prediction_2d_XGB, cmap=cmap_pred)
        axs[i, 4].set_title('XGB Prediction (ROI {})'.format(roi))
    if 'best_knn_model' in locals():
        axs[i, 5].imshow(input_prediction_2d_KNN, cmap=cmap_pred)
        axs[i, 5].set_title('KNN Prediction (ROI {})'.format(roi))
   
#Create custom cmap for legend
colors_legend= ['#000000', '#ff0000', '#00ff00', '#0000ff']
cmap_legend = ListedColormap(colors_legend)

#Create legend 
labels_legend = ["Unlabelled", "Target_vegetation", "Other_vegetation", "Non_vegetation"]
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors_legend]
plt.legend(handles, labels_legend, bbox_to_anchor=(1.87,7.03),fontsize=10)

plt.savefig('validation model outcomes/Prediction_validation.png', bbox_inches='tight')
plt.show()

print("Displayed and exported")
#------------------------------------------------------------------#

# After selecting the best model, exporting the prediction results for selected model

print('Exporting the prediction results... ', end="", flush=True)

# Define the custom colormaps for masks
colors_mask = ['#000000', '#ff0000', '#00ff00', '#0000ff']
cmap_mask = ListedColormap(colors_mask)

# Define the custom colormaps for predictions
colors_pred = ['#ff0000', '#00ff00', '#0000ff']
cmap_pred = ListedColormap(colors_pred)

# Plotting the input image, mask and prediction for all the models
rois = [1,2,3,4,5,6,7,8,9,10,11]
for i, roi in enumerate(rois):
    input_img_file = 'C:/Users/N10837647/OneDrive - Queensland University of Technology/Desktop/nz_hw_rs/Foliage_classification/rois/nz_hw_roi__{}.tif'.format(roi)
    ds_img = gdal.Open(input_img_file)
    input_img = np.array([ds_img.GetRasterBand(j+1).ReadAsArray() for j in range(5)])
    input_img = np.transpose(input_img, (1,2,0))
    input_img = exposure.equalize_hist(input_img)
    veg_indices = calculate_veg_indices(input_img)
    input_img = np.concatenate((input_img, veg_indices), axis=2)
    input_img_hist_array = input_img[np.newaxis, ...]
    input_img_hist_array_2d = input_img_hist_array.reshape(-1, input_img_hist_array.shape[-1])

    
    input_prediction_XGB = best_xgb_model.predict(input_img_hist_array_2d)
    input_prediction_2d_XGB = input_prediction_XGB.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
    plt.imshow(input_prediction_2d_XGB, cmap=cmap_pred)
    plt.axis('off')
    plt.savefig('selected_model_prediction/XGB_predicted_ROI_{}.png'.format(roi), dpi=300, bbox_inches='tight')
print("Exported")
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#




