# Import general python libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import convolve

import tensorflow as tf

# Import the GDAL module from the osgeo package
from osgeo import gdal

# Import necessary functions from scikit-learn
from sklearn.metrics import confusion_matrix, classification_report

# Import necessary functions and classes from Keras
from keras.models import load_model
#----------------------------------------------------------------------#
# Set the parameters to control the operations
apply_veg_indices = True 
apply_gaussian = False
apply_mean = False
apply_convolution=False
delete_bands = False

# Specify the bands to be deleted
deleted_bands = [0,1,2,3,4] if delete_bands else None  # Adjust this list based on your scenario

# Minimum width and height for filtering
min_width = 0
min_height = 0
max_width = 20000
max_height = 20000

tile_size = 128
overlap_percentage = 0.3
test_size=0.25

learning_rate=0.001
batch_size=25    
epochs=75

n_classes = 4
target_names = ['BB','OV','NV'] 
#----------------------------------------------------------------------#
def post_idx_calc(index, normalise):
    # Replace nan with zero and inf with finite numbers
    idx = np.nan_to_num(index)
    if normalise:
        return cv2.normalize(
            idx, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        return idx

# Define function to calculate vegetation indices
def calculate_veg_indices(input_img):
# Extract the all channels from the input image
    RedEdge = input_img[:, :, 3]
    nir = input_img[:, :, 4]
    red = input_img[:, :, 2]
    green = input_img[:, :, 1]
    blue = input_img[:, :, 0]

    # Calculate vegetation indices
    ndvi = (nir - red) / (nir + red)
    gndvi = (nir - green) / (nir + green)
    ndre = (nir - RedEdge) / (nir + RedEdge)
    gci = (nir)/(green) - 1
    msavi = ((2 * nir) + 1 -(np.sqrt(np.power((2 * nir + 1), 2) - 8*(nir - red))))/2
    exg = ((2*green)-red-blue)/(red+green+blue)
    sri = (nir / red)
    arvi = (nir - (2*red - blue)) / (nir + (2*red - blue))
    lci = (nir - RedEdge) / (nir + red)
    hrfi = (red - blue) / (green + blue)
    dvi = (nir - red)
    rvi = (nir)/(red)
    tvi = (60*(nir - green)) - (100 * (red - green))
    gdvi = (nir - green)
    ngrdi = (green - red) / (green + red)
    grvi = (red - green) / (red + green)
    rgi = (red / green)
    endvi = ((nir + green) - (2 * blue)) / ((nir + green) + (2 * blue))
    evi=(2.5 * (nir - red)) / (nir + (6 * red) - (7.5 * blue) + 1)
    sipi= (nir - blue) / (nir - red)
    osavi= (1.16 * (nir - red)) / (nir + red + 0.16)
    gosavi=(nir - green) / (nir + green + 0.16)
    exr= ((1.4 * red) - green) / (red + green + blue)
    exgr= (((2 * green) - red - blue) / (red + green + blue)) - (((1.4 * red) - green) / (red + green + blue))
    ndi=(green - red) / (green + red)
    gcc= green / (red + green + blue)
    reci= (nir) / (RedEdge) - 1
    ndwi= (green - nir) / (green + nir)

    #veg_indices = np.stack((ndvi,ndre,hrfi,gndvi,gci,msavi,exg,sri,arvi,lci, dvi, rvi, tvi, gdvi, ngrdi, grvi, rgi, endvi, evi,sipi,osavi,gosavi,exr,exgr,ndi,gcc,reci,ndwi), axis=2)
    veg_indices = np.stack((dvi,msavi,gdvi,rgi,gndvi), axis=2)

    return veg_indices
#----------------------------------------------------------------------#
# Define a 7x7 low-pass averaging kernel
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

# Define a function to apply Gaussian blur to an image
def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)

# Function to apply mean filter in a 3x3 window
def apply_mean_filter(img):
    return cv2.blur(img, (3,3))
#----------------------------------------------------------------------#
# Define the tile size and overlap percentage
tile_size = tile_size
overlap = int(tile_size * overlap_percentage)
#----------------------------------------------------------------------#
# Define the root directory with input images and respective masks
root_image_folder = r'/home/n10837647/hpc/csu/23052024_all_model/bb'

# Count the number of vegetation indices only when apply_veg_indices is True
num_veg_indices = calculate_veg_indices(np.zeros((1, 1, 27))).shape[2] if apply_veg_indices else 0

# Define a configuration string based on the parameter values
config_str = (
    f'tile_[{tile_size}]_o.lap_[{overlap_percentage}]_t.size_[{test_size}]_'
    f'b.size_[{batch_size}]_epochs_[{epochs}]_vis_[{apply_veg_indices}]_num_vi_[{num_veg_indices}]_'
    f'gau_[{apply_gaussian}]_mean_[{apply_mean}]_con_[{apply_convolution}]_'
    f'd._bands_{deleted_bands}_l.rate_[{learning_rate}]'
) if delete_bands else (
    f'tile_[{tile_size}]_o.lap_[{overlap_percentage}]_t.size_[{test_size}]_'
    f'b.size_[{batch_size}]_epochs_[{epochs}]_vis_[{apply_veg_indices}]_num_vi_[{num_veg_indices}]_'
    f'gau_[{apply_gaussian}]_mean_[{apply_mean}]_con_[{apply_convolution}]_del_band[false]_l.rate_[{learning_rate}]'
)

root_model_folder = os.path.join(root_image_folder, f'unet_model&outcomes_{config_str}')
#----------------------------------------------------------------------#
# Load unet model
unet_model = load_model(os.path.join(root_model_folder,'unet_save_best_model.hdf5'))
print("Model loaded")
#----------------------------------------------------------------------#
# Store the tiled images and masks
image_patches = []
mask_patches = []

# Define a function to get the width and height of an image using GDAL
def get_image_dimensions(file_path):
    ds = gdal.Open(file_path)
    if ds is not None:
        width = ds.RasterXSize
        height = ds.RasterYSize
        return width, height
    return None, None

# Specify the folder paths for images and masks
image_folder_path = os.path.join(root_image_folder, 'msi_rois/testing')
mask_folder_path = os.path.join(root_image_folder, 'mask_rois/testing')

# Minimum width and height for filtering
min_width = 0
min_height = 0
max_width = 20000
max_height = 20000

# Filter image and mask files based on dimensions
filtered_image_files = []
filtered_mask_files = []

input_img_folder = os.path.join(root_image_folder, 'msi_rois/testing')
input_mask_folder = os.path.join(root_image_folder, 'mask_rois/testing')

img_files = [file for file in os.listdir(input_img_folder) if file.endswith(".tif")]
mask_files = [file for file in os.listdir(input_mask_folder) if file.endswith(".tif")]

# Iterate through the image files
for img_file in img_files:
    img_path = os.path.join(image_folder_path, img_file)
    img_width, img_height = get_image_dimensions(img_path)
    
    if img_width is not None and img_height is not None:
        if min_width <= img_width <= max_width and min_height <= img_height <= max_height:
            filtered_image_files.append(img_path)

# Iterate through the mask files
for mask_file in mask_files:
    mask_path = os.path.join(mask_folder_path, mask_file)
    mask_width, mask_height = get_image_dimensions(mask_path)
    
    if mask_width is not None and mask_height is not None:
        if min_width <= mask_width <= max_width and min_height <= mask_height <= max_height:
            filtered_mask_files.append(mask_path)

# Print the number of filtered image and mask files
print(f"Number of filtered image files: {len(filtered_image_files)}")
print(f"Number of filtered mask files: {len(filtered_mask_files)}")

# Sort the filtered files to ensure consistent ordering
filtered_image_files.sort()
filtered_mask_files.sort()

for i in range(len(filtered_image_files)):
    img_file = os.path.basename(filtered_image_files[i])  # Get the file name without the path
    mask_file = os.path.basename(filtered_mask_files[i])  # Get the file name without the path
    
    ds_img = gdal.Open(filtered_image_files[i])
    ds_mask = gdal.Open(filtered_mask_files[i])
    width = ds_img.RasterXSize
    height = ds_img.RasterYSize

    # Calculate the number of tiles in the image
    num_tiles_x = (width - tile_size) // (tile_size - overlap) + 1
    num_tiles_y = (height - tile_size) // (tile_size - overlap) + 1

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate the tile coordinates
            x_start = x * (tile_size - overlap)
            y_start = y * (tile_size - overlap)
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Extract the image tile
            input_bands = 5  # Number of input bands
            input_img = np.array([ds_img.GetRasterBand(j + 1).ReadAsArray(x_start, y_start, tile_size, tile_size) for j in range(input_bands)])
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img = exposure.equalize_hist(input_img)
            
            if apply_veg_indices:
                veg_indices = calculate_veg_indices(input_img)
                input_img = np.concatenate((input_img, veg_indices), axis=2)
            
            if delete_bands:
                input_img = np.delete(input_img, deleted_bands, axis=2)

            if apply_convolution:
                for c in range(input_img.shape[2]):
                    input_img[:, :, c] = convolve(input_img[:, :, c], kernel)
            
            if apply_gaussian:
                input_img = apply_gaussian_blur(input_img)

            if apply_mean:
                input_img = apply_mean_filter(input_img)

            input_mask = ds_mask.GetRasterBand(1).ReadAsArray(x_start, y_start, tile_size, tile_size).astype(int)
           
            image_patches.append(input_img)
            mask_patches.append(input_mask)

    print(f"Processed image: {img_file} --> Processed mask: {mask_file}")

# Convert the lists to NumPy arrays
image_patches = np.array(image_patches)
mask_patches = np.array(mask_patches)
#----------------------------------------------------------------------#
# Print the shape of the arrays
print("image_patches.shape: {}".format(image_patches.shape))
print("mask_patches.shape: {}".format(mask_patches.shape))

output_file = os.path.join(root_model_folder, 'unet_testing_samples.txt')
# Save the print results to a text file
with open(output_file, "w") as file:
    file.write("image_patches.shape: {}\n".format(image_patches.shape))
    file.write("mask_patches.shape: {}\n".format(mask_patches.shape))
#----------------------------------------------------------------------#
#Confusion_matrix and Classification_report
#----------------#
# Confusion_matrix
#----------------#
#Create the mask

mask = (mask_patches != 0)

# Apply the mask to ignore class -1
y_test_mask = mask_patches[mask]

# Predict on the test data
y_pred = unet_model.predict(image_patches)

y_pred_classes = np.argmax(y_pred, axis=-1)

# Apply the mask to ignore class -1 in y_pred
y_pred_mask = y_pred_classes[mask]

# Calculate the confusion matrix
cm = confusion_matrix(y_test_mask.flatten(), y_pred_mask.flatten())

print("Confusion Matrix:")
print(cm)

# #------------------------------------------------------------------#
# Plot the confusion matrix 

# Plot the confusion matrix using heatmap()
plt.figure()
sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.title('confusion matrix_heatmap')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig(os.path.join(root_model_folder, 'unet_cm_heatmap_testing.png'), bbox_inches='tight')
plt.show()
print('Saved confusion matrix_heatmap')
#------------------------------------------------------------------#
#---------------------#
# classification report
#---------------------#
cr = classification_report(y_test_mask.flatten(), y_pred_mask.flatten(), target_names=target_names)

# Print the classification report
print(cr)
#------------------------------------------------------------------#
# Export confusion matrix and classification report as .txt
file_path = os.path.join(root_model_folder, 'unet_model_testing_performance_report.txt')
with open(file_path, 'w') as file:
    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\n")
    file.write("Classification Report:\n")
    file.write(cr)
print('Saved classification_and_confusion_report')
#------------------------------------------------------------------#
#IOU
# Calculate and save IoU for each class
class_iou = []
with open(file_path, 'a') as file:
    file.write("\n\nIoU Results:\n")
    for i in range(5):
        true_class = (y_test_mask.flatten() == i)
        pred_class = (y_pred_mask.flatten() == i)
        intersection = np.sum(true_class * pred_class)
        union = np.sum(true_class) + np.sum(pred_class) - intersection
        iou = intersection / union
        class_iou.append(iou)
        file.write("IoU for class {}: {:.2f}\n".format(i+1, iou))
        print("IoU for class {}: {:.2f}".format(i+1, iou))
# Calculate and save average IoU
average_iou = np.mean(class_iou)
with open(file_path, 'a') as file:
    file.write("Average IoU: {:.2f}".format(average_iou))
    print("Average IoU: {:.2f}".format(average_iou))
print('Saved IoU results')
#-------------------------xxxxxx---------------------------------------#