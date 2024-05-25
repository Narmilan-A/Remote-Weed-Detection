import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# File paths
image_path = 'F:/csu_contract_narmilan_ongoing/BitouBush/hsi_spectral/hsi_roi_1.tif'
mask_path = 'F:/csu_contract_narmilan_ongoing/BitouBush/hsi_spectral/mask_1.tif'

# Open the image and mask
image_ds = gdal.Open(image_path)
mask_ds = gdal.Open(mask_path)

# Convert image and mask to numpy arrays
image = np.array([image_ds.GetRasterBand(i + 1).ReadAsArray() for i in range(image_ds.RasterCount)])
mask = mask_ds.GetRasterBand(1).ReadAsArray()

# Class names
class_names = {1: 'Bitou Bush', 2: 'Other vegetation', 3: 'Non vegetation'}

# Initialize empty arrays to store spectral values
spectral_values = {class_id: [] for class_id in class_names}

# Iterate through each band and compute mean spectral values for each class
for band_idx in range(image.shape[0]):
    for class_id in class_names:
        class_pixels = image[band_idx][mask == class_id]
        mean_spectral_value = np.mean(class_pixels)
        spectral_values[class_id].append(mean_spectral_value)

# Calculate the absolute differences between class curves for each band
differences = {}
for i, class_id1 in enumerate(class_names):
    for j, class_id2 in enumerate(class_names):
        if i < j:
            diff = np.abs(np.array(spectral_values[class_id1]) - np.array(spectral_values[class_id2]))
            differences[(class_id1, class_id2)] = diff

# Get the number of bands
num_bands = image.shape[0]

# Plot only the bands
plt.figure(figsize=(10, 6))
x_positions = np.arange(1, num_bands + 1)
for class_id in class_names:
    plt.plot(x_positions, spectral_values[class_id][:num_bands], label=f' {class_names[class_id]}')
plt.xlabel('Band Number')
plt.ylabel('Mean Spectral Value')
plt.title('Spectral Signature Curves - Bands Only')
plt.xticks(x_positions)
# Update the x-axis ticks for every 50 bands
x_ticks = np.arange(1, num_bands + 1, 25)
plt.xticks(x_ticks)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('spectral_signature_plot.png', dpi=300)
plt.show()
#---------------------------------------------------------------------------------------------#
# Select top bands
# Define the class IDs and their pairs to compare
class_pairs = [(1, 2), (1, 3), (2, 3)]

# Define the number of top items to display for bands
top_items_count_bands = 100

# Specify the output file paths
output_file = "spectral_difference_results.txt"
all_selected_bands_output_file = "all_selected_bands.txt"

# Initialize a set to store selected bands from all class pairs
selected_bands_set = set()

with open(output_file, "w") as f, open(all_selected_bands_output_file, "w") as bands_f:
    # Calculate the absolute differences between class curves for each band
    for idx, (class_one, class_another) in enumerate(class_pairs):
        band_differences = {}

        for band_idx in range(num_bands):
            diff = np.abs(spectral_values[class_one][band_idx] - spectral_values[class_another][band_idx])
            band_differences[band_idx + 1] = diff  # Add +1 to adjust the band index


        # Find the top items with the highest differences
        top_band_differences = sorted(band_differences.items(), key=lambda x: x[1], reverse=True)[:top_items_count_bands]
        top_bands = [band[0] for band in top_band_differences]

        # Update the selected bands set with the current class pair's top bands
        selected_bands_set.update(top_bands)

        result_dict = {
            f"Class {class_one} vs {class_another}": top_bands
        }

        result_message = f"Top {top_items_count_bands} bands with the most significant spectral difference between classes {class_one} and {class_another}:\n"
        print(result_message)
        f.write(result_message)

        result_info = str(result_dict) + "\n"
        print(result_info, end="")
        f.write(result_info)

        f.write("\n")

        confirmation_message = f"Results saved to {output_file}\n"
        print(confirmation_message)
        f.write(confirmation_message)

    selected_bands_info = "[" + ", ".join(map(str, [band for band in selected_bands_set])) + "]\n"
    print(selected_bands_info, end="")
    bands_f.write(selected_bands_info)
#-------------------------------------------------------------------------#
# Plot only the bands with annotation of selected bands
plt.figure(figsize=(10, 6))
x_positions = np.arange(1, num_bands + 1)

# Plot spectral signature curves for each class
for class_id in class_names:
    plt.plot(x_positions, spectral_values[class_id][:num_bands], label=f'{class_names[class_id]} ({class_id})')

# Define the class IDs and their pairs to compare
class_pairs = [(1, 2), (1, 3), (2, 3)]

# Define the number of top items to display for bands
top_items_count_bands = 100

# Get a colormap for the class pairs
colormap = plt.cm.get_cmap('prism', len(class_pairs))

# Calculate the absolute differences between class curves for each band
for idx, (class_one, class_another) in enumerate(class_pairs):
    band_differences = {}

    for band_idx in range(num_bands):
        diff = np.abs(spectral_values[class_one][band_idx] - spectral_values[class_another][band_idx])
        band_differences[band_idx + 1] = diff

    # Find the top items with the highest differences
    top_band_differences = sorted(band_differences.items(), key=lambda x: x[1], reverse=True)[:top_items_count_bands]
    top_bands = [band[0] for band in top_band_differences]

    # Get a color from the colormap based on the index
    color = colormap(idx)
    
    plt.scatter(top_bands, [spectral_values[class_one][band_idx - 1] for band_idx in top_bands],
                color=color, label=f'Selected Bands {class_one} vs {class_another}', marker='o')

plt.xlabel('Band Number')
plt.ylabel('Mean Spectral Value')
plt.title('Spectral Signature Curves')
# Set the x-axis ticks with an interval of 25, starting from 1
plt.xticks(np.arange(1, num_bands + 1, 25))
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the plot with 300 dpi
plt.savefig('spectral_signature_top_plot.png', dpi=300)
# Display the plot
plt.show()
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#





