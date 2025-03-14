
import rasterio
import os
import pandas as pd
import numpy as np
import albumentations as A
import cv2
from tqdm import tqdm


def read_numpy(path_img, path_mask):
    with rasterio.open(path_img) as src_img:
        meta_img = src_img.profile.copy()
        img =  src_img.read()
        img = img.transpose(1,2,0) # original 8x256x256 --> 256x256x8 channel last

    with rasterio.open(path_mask) as src_mask:
        meta_mask = src_mask.profile.copy()
        mask = src_mask.read()
        mask = mask.transpose(1, 2, 0)

    return {"img": img, "meta_img":meta_img, "mask": mask, "meta_mask":meta_mask}

class DataAugmentation:
    def __init__(self,path_df_a:pd.DataFrame, path_df_b:pd.DataFrame, save_folder:str, p:float):
        self.transform = A.Compose([A.RandomResizedCrop(size = (256,256), scale=(0.7, 1.0), interpolation = cv2.INTER_NEAREST, mask_interpolation = cv2.INTER_NEAREST,  p=p),
                                    A.HorizontalFlip(p=p),
                                    A.Rotate(limit=(-10, 10), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=p),
                                    A.Affine(shear={"x": 0, "y": (-10,10)}, p=p),
                                    ])
        
        self.df_a = pd.read_csv(path_df_a, header = 0, sep = ",")
        self.df_b = pd.read_csv(path_df_b, header = 0 , sep = ",")

        self.save_folder = save_folder

        # Generator
        np.random.seed(42)

    def calc_length(self):
        diff = np.abs(len(self.df_a ) - len(self.df_b))

        return diff

    def get_df(self):
        if len(self.df_a) > len(self.df_b):
            fill_domain = self.df_b
        else:
            fill_domain = self.df_a
        
        return fill_domain

    def file_name(self, path_img, path_mask):
        name_img = os.path.basename(path_img)
        name_mask = os.path.basename(path_mask)

        name_img = os.path.splitext(name_img)[0]
        name_mask = os.path.splitext(name_mask)[0]

        return name_img, name_mask

    def make_aug(self):
        diff  = self.calc_length()
        print(f"{diff} number of samples wil be crated with data augmentaion")
        df= self.get_df()

        length_df = len(df)

        for i in tqdm(range(diff)):
            df_row = df.iloc[i%length_df]
            img_path = df_row["SAR_Paths"]
            mask_path = df_row["Mask_Paths"]

            dic = read_numpy(img_path, mask_path)
            img = dic["img"]
            mask = dic["mask"]

            assert img.shape == np.zeros(shape = (256,256,8)).shape,f"Error in img {img_path}, got shape {img.shape}"
            assert mask.shape == np.zeros(shape = (256,256,1)).shape,f"Error in Mask {mask_path}, got shape {mask.shape}"

            transformed = self.transform(image = img, mask = mask)

            img = transformed["image"]
            mask = transformed["mask"]

            #create unique path
            name_img, name_mask = self.file_name(img_path, mask_path) # e.g. 20160419_Houston_ID_1_4_SAR

            unique_name_img = f"{name_img}_aug_{i}.tif"
            unique_name_mask = f"{name_mask}_aug_{i}.tif"

            save_folder_img = os.path.join(self.save_folder + "/SAR/")
            save_folder_mask = os.path.join(self.save_folder + "/GT/")

            os.makedirs(save_folder_img, exist_ok = True)
            os.makedirs(save_folder_mask, exist_ok = True)

            # update profile for img and mask
            img_profile = dic["meta_img"]
            mask_profile = dic["meta_mask"]

            # change back to numpy array and channel first
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            img = img.transpose(2,0,1) # 1,2,0
            
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            mask = mask.transpose(2,0,1)

            with rasterio.open(os.path.join(save_folder_img + unique_name_img), "w", **img_profile) as dst_img:
                dst_img.write(img)

            with rasterio.open(os.path.join(save_folder_mask + unique_name_mask), "w", **mask_profile) as dst_mask:
                dst_mask.write(mask)















