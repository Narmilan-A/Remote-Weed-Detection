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
from matplotlib.image import imsave
from sklearn.decomposition import PCA
from scipy.ndimage import convolve
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
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

input_imgs = []
input_masks = []

rois = [1]
for i in rois:
    print('Importing the rois and masks ... ', end="", flush=True)
    input_img_file = '/home/n10837647/hsi/retrain_13.08.2023/rois/hsi_roi_{}.tif'.format(i)
    input_mask_file = '/home/n10837647/hsi/retrain_13.08.2023/masks/mask_{}.tif'.format(i)
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
    input_imgs.append(selected_input_img)
    input_masks.append(input_mask)

    print(f'Processed ROI {i} and its mask.')

print('Importing the rois and masks completed.')
#----------------------------------------------------------------------#
# Preprocess the data
print('Preprocess the data ... ', end="", flush=True)
X = []
y = []
for i in range(len(rois)):
    # Filtering unlabelled data
    gt_mask = ((input_masks[i] > 0))
    # Filter unlabelled data from the source image and store their values in the 'X' features variable
    x_array = input_imgs[i][gt_mask, :]
    # Select only labelled data from the labelled image and store their values in the 'y' labels variable
    y_array = input_masks[i][gt_mask]-1
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
print('Resampling of dataset ... ', end="", flush=True)
# Separate the three classes
id_1 = np.where(y_train == 0)[0]
id_2 = np.where(y_train == 1)[0]
id_3 = np.where(y_train == 2)[0]

# Print the pixels numbers
print("Number of pixels in id_1: :"f"{len(id_1)}")
print("Number of pixels in id_2: :"f"{len(id_2)}")
print("Number of pixels in id_3: :"f"{len(id_3)}")

# Resample : Downsample / upsample the class
id_1_upsampled = resample(id_1, replace=True, n_samples=len(id_3), random_state=42)
id_2_upsampled = resample(id_2, replace=True, n_samples=len(id_3), random_state=42)

# Print the pixels numbers after resampling
print("Number of pixels in id_1 after resampling: :"f"{len(id_1_upsampled)}")
print("Number of pixels in id_2 after resampling: :"f"{len(id_2_upsampled)}")
print("Number of pixels in id_3 after resampling: :"f"{len(id_3)}")

# Combine the three classes into a single dataset
X = np.concatenate((X_train[id_1_upsampled], X_train[id_2_upsampled], X_train[id_3]), axis=0)
y = np.concatenate((y_train[id_1_upsampled], y_train[id_2_upsampled], y_train[id_3]), axis=0)

# Shuffle the combined dataset
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X_train = X[shuffle_idx]
y_train = y[shuffle_idx]
#----------------------------------------------------------------------#
# Load the saved model
best_knn_model = joblib.load('best_knn_model.pkl')
best_rf_model = joblib.load('best_rf_model.pkl')
best_svm_model = joblib.load('best_svm_model.pkl')
best_xgb_model = joblib.load('best_xgb_model.pkl')
#------------------------------------------------------------------#
# # List of models and their corresponding classifiers
# print('Estmatng the model performances... ', end="", flush=True)

models = ['RF', 'SVM', 'XGB', 'KNN']
classifiers = [best_rf_model, best_svm_model, best_xgb_model, best_knn_model]
labels = ["Target_vegetation", "Other_vegetation", "Non_vegetation"]

#create a folder to save the output files
import os
if not os.path.exists('validation model outcomes'):
    os.makedirs('validation model outcomes')

#create an empty string to store all the classification reports
all_cr = ''

#create an empty list to store all the confusion matrices
all_cm = []

#Loop through each model
for i, model in enumerate(models):
    print(f'Calculating metrics for {model} ...')
    
    #Predicting X_test using the current model
    y_pred = classifiers[i].predict(X_train)
    
    #Calculate the confusion matrix for the current model
    cm = confusion_matrix(y_train, y_pred)
    print(f'Confusion matrix of {model}:')
    print(cm)
    
    #Save the confusion matrix to the all_cm list
    all_cm.append(cm)
    
    #Plot the confusion matrix for the current model using heatmap()
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion matrix heatmap of {model}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f'validation model outcomes/CM_heatmap_{model}.png', bbox_inches='tight')
    plt.show()
    
   # Develop classification report only using class id=1,2,3
    cr = classification_report(y_train, y_pred, target_names=['Target_Vegetation', 'Other_Vegetation', 'Non_vegetation'])
    print(f'Classification report of {model}:')
    print(cr)
    
    #Append the classification report to the all_cr string
    all_cr += f'Classification report of {model}:\n{cr}\n\n'

#Save all the classification reports and confusion matrices in one text file
with open('validation model outcomes/validation_all_CR_CM.txt', 'w') as f:
    
    #Write the classification reports to the file
    f.write('--- Classification Reports ---\n\n')
    f.write(all_cr)
    
    #Write the confusion matrices to the file
    f.write('\n\n--- Confusion Matrices ---\n\n')
    for i, model in enumerate(models):
        f.write(f'Confusion matrix of {model}:\n{all_cm[i]}\n\n')

print("Estimated")
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#
#-------------------------------------****************************-----------------------------------------------#




