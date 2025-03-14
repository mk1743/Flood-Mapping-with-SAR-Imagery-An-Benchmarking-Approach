
from torch.utils.data import Dataset
import torch
from Utils.utils import read_c_last
import numpy as np

class UrbanSARFloods(Dataset):
    def __init__(self, df, split):
        self.split = split
        self.df = df # pandas data frame with paths
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        example = {}
        
        df_row = self.df.iloc[index]
        image_path = df_row["SAR_Paths"]
        mask_path = df_row["Mask_Paths"]
        img, mask = read_c_last(image_path, mask_path, channel_last=False) # read image and mask im channel first format for pytorch tensor

        assert img.shape == np.zeros(shape=(8,256,256)).shape,f"Error in Dataset Class IMG has not the required shape, got {img.shape}"

        # generate weights for WeightedRandomSampler

        # training or validation
        # load ground truth flood mask
        flood_mask = mask

        img = torch.from_numpy(img) # 8,256,256
        flood_mask = torch.from_numpy(flood_mask)

        example["image"] = img
        example["mask"] = flood_mask
            
        return example 
