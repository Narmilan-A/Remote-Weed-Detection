#----------------------------------------------------------------------#
# Import general python libraries
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import joblib
from matplotlib.colors import ListedColormap
from skimage import exposure 
from matplotlib.image import imsave
from scipy.ndimage import convolve
#----------------------------------------------------------------------#
#Load the saved model
classifier_KNN = joblib.load('/home/n10837647/hsi/retrain_13.08.2023/best_knn_model.pkl')
classifier_RF = joblib.load('/home/n10837647/hsi/retrain_13.08.2023/best_rf_model.pkl')
classifier_SVC = joblib.load('/home/n10837647/hsi/retrain_13.08.2023/best_svm_model.pkl')
classifier_XGB = joblib.load('/home/n10837647/hsi/retrain_13.08.2023/best_xgb_model.pkl')
#----------------------------------------------------------------------#
# Display and export the prediction results
print('Displaying and exporting the prediction results... ', end="", flush=True)
# Define the custom colormaps for masks
colors_mask = ['#000000', '#ff0000', '#00ff00', '#0000ff']
cmap_mask = ListedColormap(colors_mask)
# Define the custom colormaps for predictions
colors_pred = ['#ff0000', '#00ff00', '#0000ff']
cmap_pred = ListedColormap(colors_pred)
#----------------------------------------------------------------------#
# Bands to select
# Specify the path to the all_selected_bands.txt file
all_selected_bands_file = "/home/n10837647/hsi/retrain_13.08.2023/all_selected_bands.txt"
# Read the content of the file
with open(all_selected_bands_file, "r") as f:
    bands_info = f.read()
# Convert the string of bands into a list
selected_bands_list = eval(bands_info)
print(selected_bands_list)
#----------------------------------------------------------------------#
# Define a 7x7 low-pass averaging kernel
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

input_images = []
input_maskss = []
input_hsi_images = []

rois = [3,10]
# Plotting the input image, mask, and prediction for all the models
fig, axs = plt.subplots(len(rois), 6, figsize=(19, 6))
for i, roi in enumerate(rois):
    print('Importing the rois and masks ... ', end="", flush=True)

    input_img_file = '/home/n10837647/hsi/retrain_13.08.2023/rois/hsi_roi_{}.tif'.format(roi)
    input_mask_file = '/home/n10837647/hsi/retrain_13.08.2023/masks/mask_{}.tif'.format(roi)
    
    ds_img = gdal.Open(input_img_file)
    ds_mask = gdal.Open(input_mask_file)
    
    # Select specific bands from input image
    selected_input_img = np.array([ds_img.GetRasterBand(band).ReadAsArray() for band in selected_bands_list])
    selected_input_img = np.transpose(selected_input_img, (1, 2, 0))
    selected_input_img = exposure.equalize_hist(selected_input_img)

    # Apply 7x7 low-pass filter using convolution
    for c in range(selected_input_img.shape[2]):
        selected_input_img[:, :, c] = convolve(selected_input_img[:, :, c], kernel)
    
    input_mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(int)
    input_images.append(selected_input_img)
    input_maskss.append(input_mask)
#---------------------------------------------------------------------------------------------------#
    all_band_input_img = np.array([ds_img.GetRasterBand(i + 1).ReadAsArray() for i in range(ds_img.RasterCount)])
    all_band_input_img = np.transpose(all_band_input_img, (1, 2, 0))
    all_band_input_img = exposure.equalize_hist(all_band_input_img)
    input_hsi_images.append(all_band_input_img)
#---------------------------------------------------------------------------------------------------#
    print(f'Processed ROI {i} and its mask.')
    input_prediction_2d_RF = None
    input_prediction_2d_SVC = None
    input_prediction_2d_XGB = None
    input_prediction_2d_KNN = None
#---------------------------------------------------------------------------------------------------#
    # Predict using the classifier models
    input_img_hist_array = selected_input_img[np.newaxis, ...]
    input_img_hist_array_2d = input_img_hist_array.reshape(-1, input_img_hist_array.shape[-1])

    if 'classifier_RF' in locals():
        input_prediction_RF = classifier_RF.predict(input_img_hist_array_2d)
        input_prediction_2d_RF = input_prediction_RF.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_RF = '/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_geoinfo/RF_hsi_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_RF = driver.Create(pred_image_file_RF, input_prediction_2d_RF.shape[1], input_prediction_2d_RF.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_RF.GetRasterBand(1).WriteArray(input_prediction_2d_RF)
        pred_image_ds_RF.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_RF.SetProjection(ds_img.GetProjection())
        imsave('/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_img/rf_pred_hsi_ROI_{}.png'.format(roi), input_prediction_2d_RF, cmap=cmap_pred, dpi=330)
        pred_image_ds_RF = None

    if 'classifier_SVC' in locals():
        input_prediction_SVC = classifier_SVC.predict(input_img_hist_array_2d)
        input_prediction_2d_SVC = input_prediction_SVC.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_SVC = '/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_geoinfo/SVC_hsi_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_SVC = driver.Create(pred_image_file_SVC, input_prediction_2d_SVC.shape[1], input_prediction_2d_SVC.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_SVC.GetRasterBand(1).WriteArray(input_prediction_2d_SVC)
        pred_image_ds_SVC.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_SVC.SetProjection(ds_img.GetProjection())
        imsave('/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_img/svm_pred_hsi_ROI_{}.png'.format(roi), input_prediction_2d_SVC, cmap=cmap_pred, dpi=330)
        pred_image_ds_SVC = None

    if 'classifier_XGB' in locals():
        input_prediction_XGB = classifier_XGB.predict(input_img_hist_array_2d)
        input_prediction_2d_XGB = input_prediction_XGB.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_XGB = '/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_geoinfo/XGB_hsi_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_XGB = driver.Create(pred_image_file_XGB, input_prediction_2d_XGB.shape[1], input_prediction_2d_XGB.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_XGB.GetRasterBand(1).WriteArray(input_prediction_2d_XGB)
        pred_image_ds_XGB.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_XGB.SetProjection(ds_img.GetProjection())
        imsave('/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_img/xgb_pred_hsi_ROI_{}.png'.format(roi), input_prediction_2d_XGB, cmap=cmap_pred, dpi=330)
        pred_image_ds_XGB = None

    if 'classifier_KNN' in locals():
        input_prediction_KNN = classifier_KNN.predict(input_img_hist_array_2d)
        input_prediction_2d_KNN = input_prediction_KNN.reshape(input_img_hist_array.shape[1],input_img_hist_array.shape[2])
        pred_image_file_KNN = '/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_geoinfo/KNN_hsi_predicted_ROI_{}.dat'.format(roi)
        driver = gdal.GetDriverByName('ENVI')
        pred_image_ds_KNN = driver.Create(pred_image_file_KNN, input_prediction_2d_KNN.shape[1], input_prediction_2d_KNN.shape[0], 1, gdal.GDT_Float32)
        pred_image_ds_KNN.GetRasterBand(1).WriteArray(input_prediction_2d_KNN)
        pred_image_ds_KNN.SetGeoTransform(ds_img.GetGeoTransform())
        pred_image_ds_KNN.SetProjection(ds_img.GetProjection())
        imsave('/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_img/knn_pred_hsi_ROI_{}.png'.format(roi), input_prediction_2d_KNN, cmap=cmap_pred, dpi=330)
        pred_image_ds_KNN = None
                                                 
    # Display the input, mask, and predicted images for the current ROI

    #if more than two rois
    #axs[i,0].imshow(all_band_input_img[:, :, [190,115,40]])
    #axs[i,0].set_title('Input Image (ROI {})'.format(roi))

    axs[i,0].imshow(all_band_input_img[:, :, [190,115,40]])
    axs[i,0].set_title('Input Image (ROI {})'.format(roi))

    axs[i,1].imshow(input_mask, cmap=cmap_mask)
    axs[i,1].set_title('Mask Image (ROI {})'.format(roi))
    
    axs[i,2].imshow(input_prediction_2d_RF, cmap=cmap_pred)
    axs[i,2].set_title('RF Prediction (ROI {})'.format(roi))

    axs[i,3].imshow(input_prediction_2d_SVC, cmap=cmap_pred)
    axs[i,3].set_title('SVM Prediction (ROI {})'.format(roi))

    axs[i,4].imshow(input_prediction_2d_XGB, cmap=cmap_pred)
    axs[i,4].set_title('XGB Prediction (ROI {})'.format(roi))

    axs[i,5].imshow(input_prediction_2d_KNN, cmap=cmap_pred)
    axs[i,5].set_title('KNN Prediction (ROI {})'.format(roi))
   
#Create custom cmap for legend
colors_legend= ['#000000', '#ff0000', '#00ff00', '#0000ff']
cmap_legend = ListedColormap(colors_legend)

#Create legend 
labels_legend = ["Unlabelled", "Target_vegetation", "Other_vegetation", "Non_vegetation"]
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors_legend]
plt.legend(handles, labels_legend, bbox_to_anchor=(1.9,2.3),fontsize=10)
plt.savefig('/home/n10837647/hsi/retrain_13.08.2023/Training model outcomes/pred_img/training_hsi_rois_prediction_1.png', bbox_inches='tight', dpi=300)

plt.show(block=False)                                                       

print("Displayed and exported")
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#
