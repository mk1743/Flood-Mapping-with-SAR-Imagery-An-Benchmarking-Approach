'''
For creation of pandas dataframe:

https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection/blob/main/src/utils/dataset_utils.py

For creation of tiles:

https://github.com/DreamPTK/FloodModel_workshop/blob/2ba554c25feee325c3b4dad18b5ac80697b27273/Material/Day%203/Excercise/Data%20Preparation/Extra%20Resources/DataPreparation.ipynb

retrieved both on 2025-02-01
'''

# Imports

import pandas as pd
from glob import glob
import os
from datetime import datetime
import rasterio
from affine import Affine
import numpy as np
from tqdm import tqdm
import subprocess

# Helper Functions 

def get_filename(filepath): # helper function for create_df
        return os.path.split(filepath)[1]

def get_dates(file_names): # helper function for crate_df
    dates = []
    for n in file_names:
        date_str = "".join(n.split("_")[0])
        date = datetime.strptime(date_str, "%Y%m%d").date()
        date = date.strftime("%Y-%m-%d")
        dates.append(date)
    return dates    


def create_df(data_dir:str, save_folder:str, domain:str, files_256:bool):
    '''
    data_dir: path to the folder with images and gt e.g. 01_NF
    save_foder: folder to the path where the csv file will be saved, e.g. /content
    domain: Name of the domain for naming the csv file, e.g. FO
    files_256: boolean value which dertrmines if set _256 --> files_256 = True to the name of the csv file or _512 --> files_256 = False
    '''
    dates = []
    data = []

    # SAR image paths

    image_paths = sorted(glob(data_dir +  "/SAR/*.tif"))

    # Label Paths
    label_paths = sorted(glob(data_dir + "/GT/*.tif"))
    
    assert image_paths != label_paths, f"Error! There must be as many SAR images as masks! I got \n Images: {len(image_paths)} \n Masks: {len(label_paths)}!"

    # Region file names
    file_names = [get_filename(pth) for pth in image_paths] # 20170830_Houston_ID_21_12_SAR.tif

    # region names
    region_names = ["".join(n.split("_")[1]) for n in file_names]

    # aquision date
    dates = get_dates(file_names)

    # generate dictionary for pandas dataframe
    for pth in image_paths:
        with rasterio.open(pth) as src:
            width, height = src.width, src.height
            data.append({
                "SAR_Paths":pth,
                "Mask_Paths":label_paths[image_paths.index(pth)],
                "Region_Name":region_names[image_paths.index(pth)],
                "Acquisition_Date":dates[image_paths.index(pth)],
                "Image_Width":width,
                "Image_Height":height
            })

    df = pd.DataFrame(data)
    
    if files_256:

        os.makedirs(save_folder, exist_ok=True)

        save_path = f"{save_folder}/{domain}_256.csv"

        df.to_csv(save_path, index=False, header=True, sep = ",")

    else: # 512 tiles
        os.makedirs(save_folder, exist_ok=True)

        save_path = f"{save_folder}/{domain}_512.csv"

        df.to_csv(save_path, index=False, header=True, sep=",")

def create_tiles_tiff(evaluate_domain:str, patch_size:int):
    '''
    output_dir: equal to save_folder from the create_df function
    domain: equal to domain from the create_df function
    drop_rate: Float number that specifies the number of NAN values from which the image should be discarded. 
    evaluate_domain: The domain to be evaluated e.g.FU
    '''

    

    delete_tfw = ["/content/testing_case_orig/20210727_Weihui/20210727_Weihui_GT.tfw", 
    "/content/testing_case_orig/20230609_NovaKakhovka/20230609_NovaKakhovka_GT.tfw",
    "/content/testing_case_orig/20230609_NovaKakhovka/20230609_NovaKakhovka_SAR.tfw",
    "/content/testing_case_orig/20231201_Jubba_1/20231201_Jubba_SAR.tfw",
    "/content/testing_case_orig/20231201_Jubba_2/20231201_Jubba_2_SAR.tfw"]

    print(f"Remove tfw files {delete_tfw}")
    
    for l in range(len(delete_tfw)):
        path = delete_tfw[l]
        subprocess.check_call(['rm', path])
        

    sar_files = ["/content/testing_case_orig/20210727_Weihui/20210727_Weihui_SAR.tif",
    "/content/testing_case_orig/20230609_NovaKakhovka/20230609_NovaKakhovka_SAR.tif",
    "/content/testing_case_orig/20231201_Jubba_1/20231201_Jubba_SAR.tif",
    "/content/testing_case_orig/20231201_Jubba_2/20231201_Jubba_2_SAR.tif"]

    sar_files = sorted(sar_files)

    gt_files = ["/content/testing_case_orig/20210727_Weihui/20210727_Weihui_GT.tif",
    "/content/testing_case_orig/20230609_NovaKakhovka/20230609_NovaKakhovka_GT.tif",
    "/content/testing_case_orig/20231201_Jubba_1/20231201_Jubba_1_GT.tif",
    "/content/testing_case_orig/20231201_Jubba_2/20231201_Jubba_2_GT.tif"]

    gt_files = sorted(gt_files)

    print(f"\n Create tiles!")
    
    total_patches_count = 0
    for i in tqdm(range(len(sar_files))): 
        sar_path = sar_files[i]
        gt_path = gt_files[i]

        with (rasterio.open(sar_path, nodata=np.nan) as img_src,
              rasterio.open(gt_path, nodata=np.nan) as mask_src):
            # Check for errors
            assert img_src.width == mask_src.width and img_src.height == mask_src.height, "Image and mask dimensions must match."

            # read Data as numpy array
            img_np = img_src.read() 
            mask_np = mask_src.read()

            # profiles

            profile_img = img_src.profile.copy()
            profile_mask = mask_src.profile.copy()

            # num bands
            num_bands_img = img_src.count
            num_bands_mask = mask_src.count

            # Nodata in the metadata
            img_nodata = img_src.nodata
            mask_nodata = mask_src.nodata

            if img_nodata != None:
                print("Image nodata:", img_nodata)
            if mask_nodata != None:
                print("Mask nodata:", mask_nodata)

            # Determine image size
            width, height = img_src.width, img_src.height # is only listed once, as the dimensions between images and masks are identical.

            # Calculate number of tiles
            n_patches_x = width // patch_size 
            n_patches_y = height // patch_size 

            for x in range(n_patches_x): 
                for y in range(n_patches_y): 
                    total_patches_count += 1
                    # For SAR images channel first, c,h,w ; e.g. 8,512,512 --> 0,1,2
                    tile_img_data = img_np[:, x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]
                    transform_img = img_src.transform * Affine.translation(x * patch_size, y * patch_size)
                    profile_img.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': transform_img,
                        'count': num_bands_img
                    })

                    # For masks images channel first, c,h,w ; e.g. 8,512,512 --> 0,1,2
                    tile_mask_data = mask_np[:, x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]
                    transform_mask = mask_src.transform * Affine.translation(x * patch_size, y * patch_size)
                    profile_mask.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': transform_mask,
                        'count': num_bands_mask
                    })

                    if np.any(np.isnan(tile_img_data)):
                        valid_pixels = tile_img_data[~np.isnan(tile_img_data)] # extrac all valid pixels values
                        nan_pixels = np.where(np.isnan(tile_img_data[0])) # nan indicies of tile_img_data
                        mean_value = valid_pixels.mean() if len(valid_pixels) > 0 else 0.
                        tile_img_data[np.isnan(tile_img_data)] = mean_value # replace nan values with wean of valid pixels

                        tile_mask_data[0][nan_pixels] = float(0) # where nan values in the SAR images it will be replaced with the label 0 no flood
                   
                    if evaluate_domain == "FO":
                        '''
                        in the test maskthere are 3 classes: 0 = NF, 1 = FO, 2 =  FU.
                        2 --> -1
                        '''
    
                        tile_mask_data[tile_mask_data == float(2)] = float(-1) # 2 to -1
                    else: # "FU"
                        '''
                        in the test maskthere are 3 classes: 0 = NF, 1 = FO, 2 =  FU.
                        1 --> -1
                        2 --> 1
                        '''
                        tile_mask_data[tile_mask_data == float(1)] = float(-1) # 1 to -1
                        tile_mask_data[tile_mask_data == float(2)] = float(1) # 2 to 1

                    
                    # check for errors
                    assert tile_img_data.shape[1] == tile_mask_data.shape[1] and tile_img_data.shape[2] == tile_mask_data.shape[2],"Patch dimensions do not match!"

                    # save tiles
                    ## paths
                    
                    sar_name = sar_path.split("/")[-2]
                    gt_name = gt_path.split("/")[-2]

                    sar_save_name = f"{sar_name}_{x}_{y}" # e.g. 20210727_Weihui; for SAR Image file name
                    gt_save_name = f"{gt_name}_{x}_{y}" # e.g. 20210727_Weihui; for SAR Image file name
                    

                    os.makedirs("/content/Test_Data_256/SAR", exist_ok = True)
                    os.makedirs("/content/Test_Data_256/GT", exist_ok = True)

                    image_tile_path = f"/content/Test_Data_256/SAR/{sar_save_name}_SAR.tif"
                    mask_tile_path = f"/content/Test_Data_256/GT/{gt_save_name}_GT.tif"

                    ## for sar images
                    with rasterio.open(image_tile_path, "w", **profile_img) as dst:
                        dst.write(tile_img_data)

                    ## for mask images
                    with rasterio.open(mask_tile_path, "w", **profile_mask) as dst:
                        dst.write(tile_mask_data)

    print(f"\n End of creating tiles!")
    
    print(f"\n Total number of patches created: {total_patches_count}")
    

if __name__ == "__main__":
    print(" -------------------------------------- BEGIN PREPROCESSING --------------------------------------")
    
    create_tiles_tiff("FO", 256)
    create_df("/content/Test_Data_256","/content",files_256=True, domain = "Test")

    #subprocess.check_call(["rm", "-r", "/content/01_NF"]) # remove original folder with 512x512 to save space 
    print(" -------------------------------------- END PREPROCESSING --------------------------------------")