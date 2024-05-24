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
from sklearn.metrics import confusion_matrix, classification_report

# Import necessary functions and classes from Keras
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Accuracy, Precision, Recall, IoU, MeanIoU, FalseNegatives, FalsePositives
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
#----------------------------------------------------------------------#
# Set the parameters to control the operations (Optional)
apply_veg_indices = True 
apply_gaussian = False
apply_mean = False
apply_convolution=False
delete_bands = False

# Specify the bands to be deleted
deleted_bands = [0,1,2,3,4] if delete_bands else None  # Adjust this list based on your scenario

# Minimum width and height for filtering (Optional)
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

n_classes = 4 # (Unlabelled(ID=0), BB: Bitou bush (ID=1), OV: Other vegetation(ID=2), NV: Non-vegetation(ID=3))
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

    # Use required Vegetation indices based on any feature extraction techniques (Optional)
    #veg_indices = np.stack((ndvi,ndre,hrfi,gndvi,gci,msavi,exg,sri,arvi,lci, dvi, rvi, tvi, gdvi, ngrdi, grvi, rgi, endvi, evi,sipi,osavi,gosavi,exr,exgr,ndi,gcc,reci,ndwi), axis=2)
    veg_indices = np.stack((dvi,msavi,gdvi,rgi,gndvi), axis=2)
 
    return veg_indices
#----------------------------------------------------------------------#
# Apply below filters based on image quality (Optional)

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
#----------------------------------------------------------------------#
#-----------------"""**Build the model**"""----------------------------#
# Import module 
# Define the number of classess (ID 0,1,2,3)

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
#----------------------------------------------------------------------#
# Model training
#----------------------------------------------------------------------#
# Define a log directory for checkpoins
log_dir = os.path.join(root_model_folder, 'log')  # Create the log directory
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#----------------------------------------------------------------------#
# specify the filepath for where to save the weights
weight_path = os.path.join(log_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
best_model_path = os.path.join(root_model_folder, 'unet_save_best_model.hdf5')
#----------------------------------------------------------------------#
# create a ModelCheckpoint for best model
checkpoint_best_model = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss', mode='min')
#----------------------------------------------------------------------#
# create a ModelCheckpoint for save weights
checkpoint_weight = ModelCheckpoint(weight_path, ave_weights_only=True, verbose=1, period=50)
#----------------------------------------------------------------------#
# Start recording time
start_time = time()

# Train the model with class weights
history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    verbose=1,
                    epochs=epochs,
                    validation_data=(X_test, y_test), 
                    callbacks=[checkpoint_best_model, checkpoint_weight],
                    shuffle=True)

# Calculate and print the training time
end_time = time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
#------------------------------------------------------------------#
mask = (y_test != 0)

# Apply the mask to ignore class 0
y_test_mask = y_test[mask]

# Predict on the test data
y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=-1)

# Apply the mask to ignore class -1 in y_pred
y_pred_mask = y_pred_classes[mask]

# Calculate the confusion matrix
cm = confusion_matrix(y_test_mask.flatten(), y_pred_mask.flatten())

print("Confusion Matrix:")
print(cm)
#------------------------------------------------------------------#
# Plot the confusion matrix 

# Plot the confusion matrix using heatmap()
plt.figure()
sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.title('confusion matrix_heatmap')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig(os.path.join(root_model_folder, 'unet_cm_heatmap_training_validation.png'), bbox_inches='tight', dpi=400)
plt.show()
print('Saved confusion matrix_heatmap')
#------------------------------------------------------------------#
#---------------------#
# classification report
#---------------------#
cr = classification_report(y_test_mask.flatten(), y_pred_mask.flatten(), target_names=target_names)

# Print the classification report
print(cr)
#----------------------------------------------------------------------#
# Export confusion matrix and classification report as .txt
file_path = os.path.join(root_model_folder, 'unet_model_training_&_validation_performance_report.txt')
with open(file_path, 'w') as file:
    file.write(f"Training Time: {training_time} seconds\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\n")
    file.write("Classification Report:\n")
    file.write(cr)
print('Saved classification_and_confusion_report')
#----------------------------------------------------------------------#
# Export training_history
# Create a DataFrame from the history
history_df = pd.DataFrame(history.history)
# Save the DataFrame to a CSV file
history_df.to_csv(os.path.join(root_model_folder,'unet_training_history.csv'), index=False)
print('Saved training_history')
#----------------------------------------------------------------------#
# plot graphs using history
num_epochs = len(history.history['loss'])
# Plot the accuracy
plt.figure(figsize=(10, 8))
plt.plot(range(1, num_epochs + 1), history.history['loss'])
plt.plot(range(1, num_epochs + 1), history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(root_model_folder, 'unet_loss_from_0.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved loss graph')
#----------------------------------------------------------------------#
# Find the index of the 25th epoch
start_epoch = 25
start_epoch_index = start_epoch - 1  # Subtract 1 to account for 0-based indexing

# Get the relevant data from the history
train_loss = history.history['loss'][start_epoch_index:]
val_loss = history.history['val_loss'][start_epoch_index:]

# Plot the loss from the 25th epoch
plt.figure(figsize=(10, 8))
plt.plot(range(start_epoch, num_epochs + 1), train_loss, label='Train')
plt.plot(range(start_epoch, num_epochs + 1), val_loss, label='Test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(root_model_folder, 'unet_loss_from_25.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved loss_from_25 graph')
#----------------------------------------------------------------------#
# Find the index of the 25th epoch
start_epoch = 50
start_epoch_index = start_epoch - 1  # Subtract 1 to account for 0-based indexing

# Get the relevant data from the history
train_loss = history.history['loss'][start_epoch_index:]
val_loss = history.history['val_loss'][start_epoch_index:]

# Plot the loss from the 25th epoch
plt.figure(figsize=(10, 8))
plt.plot(range(start_epoch, num_epochs + 1), train_loss, label='Train')
plt.plot(range(start_epoch, num_epochs + 1), val_loss, label='Test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(root_model_folder, 'unet_loss_from_50.png'), bbox_inches='tight')
plt.tight_layout()
plt.show()
print('Saved loss_from_50 graph')
#----------------------------------------------------------------------#
#IOU
# Calculate and save IoU for each class
class_iou = []
with open(file_path, 'a') as file:
    file.write("\n\nIoU Results:\n")
    for i in range(n_classes):
        true_class = (y_test_mask == i)
        pred_class = (y_pred_mask == i)
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