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
import pickle
from Config.train.FO_config import cfg # For pre-processing, it does not matter which config file is used

from Data_Augmentation.Data_Augmentation import DataAugmentation
from Utils.utils_train import give_class_weights

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
'''
def check_consistency(path_df:str, output_dir:str, domain:str):
     

    Check if the mask has more than the desired labels e.g. for non flood only label 0. If not they will be replaced to the disred label.
    
    0 = NF
    1 = FO
    2 = FU
    

    
    # make new dataframe with path to 256x256 files 
       
    df = pd.read_csv(f"{path_df}/{domain}_256.csv", header = 0, sep=",")

    
    for i in range(len(df)):
    
        df_row = df.iloc[i]
        mask_path = df_row["Mask_Paths"]
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            
            if domain == "NF": # label = 0; permitted labels only 0
            
                if np.any(mask == float(2)) or np.any(mask == float(1)):
            
                    meta = src.meta
        
                    mask = np.where(mask == float(2), float(0), mask)
                    
                    mask = np.where(mask == float(1), float(0), mask)
        
                    path = os.path.join(output_dir + "/GT_new" )
        
                    name = i.split("/")[-1].split(".")[0]
        
                    os.makedirs(path, exist_ok=True)
        
                    with rasterio.open(os.path.join(path + "/" + name + ".tif"), "w", **meta) as dst:
                        dst.write(mask, 1)
                        
            
            if domain == "FO": # label = 1; permitted labels 0 and 1
            
                if np.any(mask == float(2)):
            
                    meta = src.meta
        
                    mask = np.where(mask == float(2), float(0), mask) # replace FU (2) to NF (0)
        
                    path = os.path.join(output_dir + "/GT_new" )
        
                    name = i.split("/")[-1].split(".")[0]
        
                    os.makedirs(path, exist_ok=True)
        
                    with rasterio.open(os.path.join(path + "/" + name + ".tif"), "w", **meta) as dst:
                        dst.write(mask, 1)
                        
            if domain == "FU": # label = 2; permitted labels 0 and 2, but this masks contain all available labels 0,1 and 2
            
                if np.any(mask == float(1)):
            
                    meta = src.meta
        
                    mask = np.where(mask == float(1), float(0), mask) # replace FO (1) to NF (0)
                    
                    mask = np.where(mask == float(2), float(1), mask) # change FU to 1 for metrics 
        
                    path = os.path.join(output_dir + "/GT_new" )
        
                    name = i.split("/")[-1].split(".")[0]
        
                    os.makedirs(path, exist_ok=True)
        
                    with rasterio.open(os.path.join(path + "/" + name + ".tif"), "w", **meta) as dst:
                        dst.write(mask, 1)
'''


def create_tiles_tiff(path_csv:str, output_dir:str, domain:str, patch_size:int):
    '''
    path_csv: path to the csv file, e.g. FU_256
    output_dir: equal to save_folder from the create_df function
    domain: equal to domain from the create_df function
    '''
        
    df = pd.read_csv(path_csv, header = 0, sep=",")

    # Create output folders
    if domain == "NF":
        os.makedirs(f"{output_dir}/01_{domain}_256/SAR", exist_ok=True)
        os.makedirs(f"{output_dir}/01_{domain}_256/GT", exist_ok=True)

    if domain == "FO":
        os.makedirs(f"{output_dir}/02_{domain}_256/SAR", exist_ok=True)
        os.makedirs(f"{output_dir}/02_{domain}_256/GT", exist_ok=True)

    if domain == "FU":
        os.makedirs(f"{output_dir}/03_{domain}_256/SAR", exist_ok=True)
        os.makedirs(f"{output_dir}/03_{domain}_256/GT", exist_ok=True)

    # Column names for the pandas Data frame
    image_path_name = "SAR_Paths"
    mask_path_name = "Mask_Paths"
    date_path_name = "Acquisition_Date"
    region_path_name = "Region_Name"

    print(f"Length DataFrame: {len(df)}")

    event_id = 0 # Initialize event_id for the very first time when the function is called
    for i in tqdm(range(len(df))): # From here processing 1 image (image level)
        df_row = df.iloc[i]
        image_path = df_row[image_path_name]
        mask_path = df_row[mask_path_name]
        date_path = df_row[date_path_name] # 2016-04-19
        event_name = df_row[region_path_name]

        date_path = datetime.strptime(date_path, "%Y-%m-%d").date()
        date_path = date_path.strftime("%Y%m%d")

        with (rasterio.open(image_path, nodata=np.nan) as img_src,
              rasterio.open(mask_path, nodata=np.nan) as mask_src):
            # Check for errors
            assert img_src.width == mask_src.width and img_src.height == mask_src.height, "Image and mask dimensions must match."

            # read Data as numpy array
            img_np = img_src.read() # all bands 8x512x512
            mask_np = mask_src.read()  # all bands (mask has only one band) 1x512x512

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
            n_patches_x = width // patch_size # e.g. 256 --> 512//256 = 2
            n_patches_y = height // patch_size # e.g. 256 --> 512//256 = 2

            '''
            X 0, Y 0
            X 0, Y 1
            X 1, Y 0
            X 1, Y 1
            '''
            for x in range(n_patches_x): # 2 times
                for y in range(n_patches_y): # 2 times 
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

                    ## Replace NAN values 
                    if np.any(np.isnan(tile_img_data)):
                        valid_pixels = tile_img_data[~np.isnan(tile_img_data)] # extrac all valid pixels values
                        nan_pixels = np.where(np.isnan(tile_img_data[0])) # nan indicies of tile_img_data
                        mean_value = valid_pixels.mean() if len(valid_pixels) > 0 else 0.
                        tile_img_data[np.isnan(tile_img_data)] = mean_value # replace nan values with wean of valid pixels

                        tile_mask_data[0][nan_pixels] = float(0) # where nan values in the SAR images it will be replaced with the label 0 no flood
                    '''
                    in the mask of FU, there are 3 classes: 0 = NF, 1 = FO, 2 FU. We are only interested in class 0 and 2. But for the binary segmentation problem,
                    only values of 0 and 1 are allowed. So we have to change: 2 to 1 and 1 to 0. 
                    '''
                    if domain == "FU":
                        tile_mask_data[tile_mask_data == float(1)] = float(0) # 1 to 0
                        tile_mask_data[tile_mask_data == float(2)] = float(1) # 2 to 1
                        
                    # check for errors
                    assert tile_img_data.shape[1] == tile_mask_data.shape[1] and tile_img_data.shape[2] == tile_mask_data.shape[2],"Patch dimensions do not match!"

                    # save tiles
                    ## paths

                    unique_filename = f"{date_path}_{event_name}_{x}_{y}_{event_id}_"

                    if domain == "NF":
                        image_tile_path = f"{output_dir}/01_{domain}_256/SAR/{unique_filename}SAR.tif"  # Unique filename for each patch
                        mask_tile_path = f"{output_dir}/01_{domain}_256/GT/{unique_filename}GT.tif" # Unique filename for each patch

                        #if os.path.exists(image_tile_path) or os.path.exists(mask_tile_path):
                            #print(f"Error: File already exists! Image Path {image_tile_path} and Mask Path {mask_tile_path}")
                            #break

                    if domain == "FO":
                        image_tile_path = f"{output_dir}/02_{domain}_256/SAR/{unique_filename}SAR.tif"  # Unique filename for each patch
                        mask_tile_path = f"{output_dir}/02_{domain}_256/GT/{unique_filename}GT.tif" # Unique filename for each patch

                    if domain == "FU":
                        image_tile_path = f"{output_dir}/03_{domain}_256/SAR/{unique_filename}SAR.tif"  # Unique filename for each patch
                        mask_tile_path = f"{output_dir}/03_{domain}_256/GT/{unique_filename}GT.tif" # Unique filename for each patch

                    ## for sar images
                    with rasterio.open(image_tile_path, "w", **profile_img) as dst:
                        dst.write(tile_img_data)

                    ## for mask images
                    with rasterio.open(mask_tile_path, "w", **profile_mask) as dst:
                        dst.write(tile_mask_data)

        # Increment event ID counter after each patch; end of the loop for i in range...
        if i < len(df) -1 and event_name != df.iloc[i+1]["Region_Name"]:
            event_id = 0
        else:
            event_id += 1
    

if __name__ == "__main__":
    print(" -------------------------------------- BEGIN PREPROCESSING --------------------------------------")

    data_aug = True

    list_domains = ["FO", "FU"]

    n = 2

    for domain in list_domains:

        print(f"\nCurrent Domain: {domain}")

        main_dir  = "/content"
        data_dir_512  =f"{main_dir}/0{n}_{domain}"
        save_folder = main_dir
        file_csv_512 = f"{save_folder}/{domain}_512.csv"
        data_dir_256 = f"{save_folder}/0{n}_{domain}_256"

        create_df(data_dir=data_dir_512, save_folder=save_folder, domain=domain, files_256=False) # with 512x512 SAR images 

        create_tiles_tiff(path_csv = file_csv_512,
        output_dir =  save_folder,
        domain = domain,
        patch_size = cfg.PREPROCESSING.PATCH_SIZE)

        create_df(data_dir=data_dir_256, save_folder=save_folder, domain=domain, files_256=True) # with 256x256 SAR images 

        n +=1

    if data_aug:

        print(f" \n Begin Data Augmentation for Domain: FU")

        data_aug = DataAugmentation(path_df_a = "/content/FO_256.csv", path_df_b = "/content/FU_256.csv", save_folder="/content/Aug", p = 1.0)
        data_aug.make_aug()

        print(f" \n End Data Augmentation for Domain: FU")

        print(f" \n Copying augmented Data into 03_FU_256 folder")

        aug_sar_data = glob("/content/Aug" + "/SAR/*.tif")
        aug_gt_data = glob("/content/Aug" + "/GT/*.tif")

        data_list = [aug_sar_data,aug_gt_data]

        for n in tqdm(range(len(data_list))):
            for file in data_list[n]:
                if n == 0:
                    subprocess.check_call(["cp", file, "/content/03_FU_256/SAR/"])
                else:
                    subprocess.check_call(["cp", file, "/content/03_FU_256/GT/"])
        
        print(f" \n Copying finished!")

        subprocess.check_call(["rm", "-r", "/content/Aug"])

        print(f"\n Folder /content/Aug removed!")

        create_df(data_dir="/content/03_FU_256", save_folder="/content", domain = "FU", files_256=True) # with added data augmentation in the folder FU

        print(f"\n New dataframe for 03_FU_256 with aug data was created!")

    n = 2

    classes = [0,1]

    for domain in list_domains:
        print(f"\n Create weights for : {domain}")
        main_dir  = "/content"
        save_folder = main_dir
        file_csv_256 = f"{save_folder}/{domain}_256.csv"

        weights = give_class_weights(path_to_df = file_csv_256, classes = classes)

        with open(os.path.join(save_folder + f"/weights_0{n}_{domain}.pkl"), 'wb') as f:
            pickle.dump(weights,f)  
        
        n +=1
    print(" -------------------------------------- END PREPROCESSING --------------------------------------")