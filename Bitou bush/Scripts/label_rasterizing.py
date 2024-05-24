# Import general python libraries
from osgeo import gdal, ogr
import os
import glob
#----------------------------------------------------------------------#
def shp_to_raster(input_shp, output_raster, reference_image):
    shp_ds = ogr.Open(input_shp)
    layer = shp_ds.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    reference_ds = gdal.Open(reference_image, gdal.GA_ReadOnly)
    geo_transform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * reference_ds.RasterXSize
    y_min = y_max + geo_transform[5] * reference_ds.RasterYSize
    x_res = reference_ds.RasterXSize
    y_res = reference_ds.RasterYSize
    output_driver = gdal.GetDriverByName('GTiff')
    output_ds = output_driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(geo_transform)
    output_ds.SetProjection(projection)
    gdal.RasterizeLayer(output_ds, [1], layer, options=["ATTRIBUTE=id"])
    shp_ds = None
    output_ds = None
    print(f"Raster '{output_raster}' created successfully.")
    return output_raster
#----------------------------------------------------------------------#
def main():
    # Define the paths
    shapefile = "/home/n10837647/hpc/ant/robbos/gt_labelling_merged_1234.shp"
    input_image_folder = "/home/n10837647/hpc/ant/robbos/input_msi_mask_rois/msi_rois/training"
    output_folder = "/home/n10837647/hpc/ant/robbos/input_msi_mask_rois/mask_rois/training"
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Get a list of all 5-band images with .tif extension
    image_list = glob.glob(os.path.join(input_image_folder, "*.tif"))
    # Loop through each image and perform the rasterization
    for image_path in image_list:
        # Set the output raster name based on the input image name
        output_raster = os.path.join(output_folder, "mask_" + os.path.splitext(os.path.basename(image_path))[0] + ".tif")
        shp_to_raster(shapefile, output_raster, image_path)
if __name__ == "__main__":
    main()
#-------------------------xxxxxx---------------------------------------#