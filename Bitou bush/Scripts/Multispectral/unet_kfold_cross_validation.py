# Import general python libraries
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import convolve
from time import time
import random
import random as python_random
import tensorflow as tf

# Import the GDAL module from the osgeo package
from osgeo import gdal

# Import necessary functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# Import necessary functions and classes from Keras
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Accuracy, Precision, Recall, IoU, MeanIoU, FalseNegatives, FalsePositives
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
#----------------------------------------------------------------------#
# Set the parameters to control the operations
apply_veg_indices = False 
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

tile_size = 64
overlap_percentage = 0.2
test_size=0.25

learning_rate=0.001
batch_size=30       
epochs=150

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
    return cv2.GaussianBlur(img, (5,5), 0)

# Function to apply mean filter in a 3x3 window
def apply_mean_filter(img):
    return cv2.blur(img, (5,5))
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

root_model_folder = os.path.join(root_image_folder, f'unet_xgb__spectral_kfold_{config_str}')

# Check if the "model&outcomes" folder exists, and create it if it doesn't
if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)
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
image_folder_path = os.path.join(root_image_folder, 'msi_rois/training')
mask_folder_path = os.path.join(root_image_folder, 'mask_rois/training')

# Filter image and mask files based on dimensions
filtered_image_files = []
filtered_mask_files = []

input_img_folder = os.path.join(root_image_folder, 'msi_rois/training')
input_mask_folder = os.path.join(root_image_folder, 'mask_rois/training')

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

# Print the shape of the arrays
print("image_patches.shape: {}".format(image_patches.shape))
print("mask_patches.shape: {}".format(mask_patches.shape))
#----------------------------------------------------------------------#
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(image_patches, mask_patches, test_size=test_size, random_state=22)
#----------------------------------------------------------------------#
# save, print and confirm the model data
output_file = os.path.join(root_model_folder, 'unet_trainng_and_validation samples.txt')
# Save the print results to a text file
with open(output_file, "w") as file:
    file.write("image_patches.shape: {}\n".format(image_patches.shape))
    file.write("mask_patches.shape: {}\n".format(mask_patches.shape))

# Save the model data to the text file
with open(output_file, "a") as file:
    file.write("\nX_train shape: {}\n".format(X_train.shape))
    file.write("X_test shape: {}\n".format(X_test.shape))
    file.write("y_train shape: {}\n".format(y_train.shape))
    file.write("y_test shape: {}\n".format(y_test.shape))
    file.write("Image height: {}\n".format(X_train.shape[1]))
    file.write("Image width: {}\n".format(X_train.shape[2]))
    file.write("Image channels: {}\n".format(X_train.shape[3]))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape[1])  
print(X_train.shape[2])  
print(X_train.shape[3]) 
#-------------------------------------------------------------------------------------------------------------#
# Apply K-Fold cross validation
print ("Applying K-Fold cross validation...")
start_time = time()

cv = KFold(n_splits=10, shuffle=True, random_state=22)

# Create a list to store the best model paths and validation loss
best_model_paths = []

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = [] #save accuracy from each fold
loss_per_fold = [] #save accuracy from each fold

for fold_no, (train, test) in enumerate(cv.split(X_train, y_train), 1):
    print('   ')
    print(f'Training for fold {fold_no} ...')

    n_classes = n_classes

    def UNet(n_classes, image_height, image_width, image_channels):
        inputs = Input((image_height, image_width, image_channels))

        seed_value = 22
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        python_random.seed(seed_value)
        
        c1 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2,2))(c1)

        c2 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2,2))(c2)

        c3 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(0.1)(c3)
        c3 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
        c3= BatchNormalization()(c3)
        p3 = MaxPooling2D((2,2))(c3)

        c4 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.1)(c4)
        c4 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D((2,2))(c4)

        c5 = Conv2D(1024, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.1)(c5)
        c5 = Conv2D(1024, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
        c5 = BatchNormalization()(c5)

        u6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding="same")(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.1)(c6)
        c6 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding="same")(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.1)(c7)
        c7 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)
        c9 = BatchNormalization()(c9)

        outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

        model = Model(inputs=inputs, outputs=outputs)
        return model
#----------------------------------------------------------------------#
    # Create the model
    image_height = X_train.shape[1]
    image_width = X_train.shape[2]
    image_channels = X_train.shape[3]
    model=UNet(n_classes=n_classes, 
                            image_height=image_height, 
                            image_width=image_width, 
                            image_channels=image_channels)
    #----------------------------------------------------------------------#
    #Complie the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class= 0), metrics=['SparseCategoricalAccuracy'])

    # Specify the filepath for where to save the weights for the best model
    best_model_path = os.path.join(root_model_folder, f'save_best_model_fold_{fold_no}.hdf5')

    # Create a ModelCheckpoint for the best model based on validation loss
    checkpoint_best_model = ModelCheckpoint(
        best_model_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Train the model
    history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    verbose=1,
                    epochs=epochs,
                    validation_data=(X_test, y_test), 
                    callbacks=[checkpoint_best_model],
                    shuffle=True)

    best_model_paths.append(best_model_path)


    # Evaluate the model - report accuracy and capture it into a list for future reporting
    scores = model.evaluate(X_test, y_test, verbose=1)
    acc_per_fold.append(scores[1] * 100)

    scores = model.evaluate(X_test, y_test, verbose=1)
    loss_per_fold.append(scores[0])

    fold_no = fold_no + 1

# Initialize a list to store fold numbers and corresponding accuracies and loss
fold_and_acc_list = [(fold_no, acc) for fold_no, acc in enumerate(acc_per_fold, 1)]
fold_and_loss_list = [(fold_no, loss) for fold_no, loss in enumerate(loss_per_fold, 1)]

# Calculate the average val_binary_accuracy
average_accuracy = sum(acc_per_fold) / len(acc_per_fold)

# Calculate the average val_binary_accuracy
average_loss = sum(loss_per_fold) / len(acc_per_fold)

for fold_no, acc in enumerate(acc_per_fold, 1):
    print(f'Fold {fold_no} val_binary_accuracy: {acc}\n"')

for fold_no, acc in enumerate(loss_per_fold, 1):
    print(f'Fold {fold_no} val_binary_loss: {acc}\n"')

print(f"Average val_binary_accuracy across all folds: {average_accuracy}\n")
print(f"Average val_binary_loss across all folds: {average_loss}\n")

# Calculate and print the training time
end_time = time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# Export confusion matrix and classification report as .txt
file_path = os.path.join(root_model_folder, 'K_Fold_outcome report.txt')
with open(file_path, 'w') as file:
    file.write(f"Training Time: {training_time} seconds\n")
    for fold_no, acc in fold_and_acc_list:
        file.write(f'Fold {fold_no} val_binary_accuracy: {acc}\n')

    for fold_no, loss in fold_and_loss_list:
        file.write(f'Fold {fold_no} val_binary_loss: {loss}\n')

    file.write(f"Average val_binary_accuracy across all folds: {average_accuracy}\n")
    file.write(f"Average val_binary_loss across all folds: {average_loss}\n")
print('K_Fold_outcome report')


# Initialize fold numbers
fold_numbers = list(range(1, len(acc_per_fold) + 1))

# Plot the accuracy
plt.figure(figsize=(10, 8))
plt.bar(fold_numbers, acc_per_fold, color='green')
plt.title('Validation Accuracy vs. Fold Number')
plt.xlabel('Fold Number')
plt.ylabel('val_accuracy(%)')
plt.xticks(fold_numbers)
plt.axhline(y=average_accuracy, color='red', linestyle='--', label=f'Average Accuracy(%): {average_accuracy:.2f}')
plt.legend(loc='upper right')
plt.savefig(os.path.join(root_model_folder, 'K_Fold_accuracy.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved K_Fold Accuracy graph')

# Create a bar chart for binary loss
plt.figure(figsize=(10, 8))
plt.bar(fold_numbers, loss_per_fold, color='blue')
plt.title('Validation Loss vs. Fold Number')
plt.xlabel('Fold Number')
plt.ylabel('val_loss')
plt.xticks(fold_numbers)
plt.axhline(y=average_loss, color='red', linestyle='--', label=f'Average Loss: {average_loss:.2f}')
plt.legend(loc='upper right')
plt.savefig(os.path.join(root_model_folder, 'K-Fold_loss.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved K-Fold_loss graph')

print ("Done K-Fold cross validation")
#----------------------------------------------------------------------#
# Calculate the standard deviation of val_binary_accuracy and val_binary_loss
accuracy_std = np.std(acc_per_fold)
loss_std = np.std(loss_per_fold)

# Print the standard deviation
print(f"Standard Deviation of val_binary_accuracy: {accuracy_std:.2f}\n")
print(f"Standard Deviation of val_binary_loss: {loss_std:.2f}\n")

with open(file_path, 'a') as file:
    file.write(f"Standard Deviation of val_binary_accuracy: {accuracy_std}\n")
    file.write(f"Standard Deviation of val_binary_loss: {average_loss}\n")

# Plot the accuracy with both average and standard deviation lines
plt.figure(figsize=(10, 8))
plt.bar(fold_numbers, acc_per_fold, color='green', label='Accuracy')
plt.axhline(y=average_accuracy, color='red', linestyle='--', label=f'Average Accuracy(%): {average_accuracy:.2f}')

#plt.errorbar(fold_numbers, acc_per_fold, yerr=accuracy_std, linestyle='None', color='black', capsize=5, label='Std Deviation')
plt.errorbar(fold_numbers, acc_per_fold, yerr=accuracy_std, linestyle='None', color='black', capsize=5, label=f'Std: {accuracy_std:.2f}')

plt.title('Validation Accuracy vs. Fold Number')
plt.xlabel('Fold Number')
plt.ylabel('val_accuracy(%)')
plt.xticks(fold_numbers)
plt.legend(loc='upper right')
plt.savefig(os.path.join(root_model_folder, 'K_Fold_accuracy_std.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved K_Fold Accuracy with std graph')

# Plot the loss with both average and standard deviation lines
plt.figure(figsize=(10, 8))
plt.bar(fold_numbers, loss_per_fold, color='blue', label='Loss')
plt.axhline(y=average_loss, color='red', linestyle='--', label=f'Average Loss: {average_loss:.2f}')
#plt.errorbar(fold_numbers, loss_per_fold, yerr=loss_std, linestyle='None', color='black', capsize=5, label='Std Deviation')
plt.errorbar(fold_numbers, loss_per_fold, yerr=loss_std, linestyle='None', color='black', capsize=5, label=f'Std: {loss_std:.2f}')

plt.title('Validation Loss vs. Fold Number')
plt.xlabel('Fold Number')
plt.ylabel('val_loss')
plt.xticks(fold_numbers)
plt.legend(loc='upper right')
plt.savefig(os.path.join(root_model_folder, 'K-Fold_loss_std.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved K-Fold_loss graph')

print("Done K-Fold cross validation with std graph")
#--------------------------------------------xxxxxx-----------------------------------------------------------#

