
'''
For creation of pandas dataframe:

https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection/blob/main/src/utils/dataset_utils.py

For creation of tiles:

https://github.com/DreamPTK/FloodModel_workshop/blob/2ba554c25feee325c3b4dad18b5ac80697b27273/Material/Day%203/Excercise/Data%20Preparation/Extra%20Resources/DataPreparation.ipynb

retrieved both on 2025-02-01
'''

import torch
from torch import Generator
import rasterio
from lightning.pytorch.callbacks import ModelCheckpoint

def create_test_dataset(number_batch):
    gen = Generator()
    tensor_list_img, tensor_list_mask = [], []
    for i in range(number_batch):
        sample_img = torch.randn(size=(8,256,256), generator=gen, dtype=torch.float32)
        sample_mask = torch.randint(0,2, (1,256,256), generator=gen, dtype=torch.float32)

        tensor_list_img.append(sample_img)
        tensor_list_mask.append(sample_mask)

    dataset_img = torch.stack(tensor_list_img, dim=0)
    dataset_mask = torch.stack(tensor_list_mask, dim=0)

    return dataset_img, dataset_mask

def read_normalize(img_path, mask_path, mean, std):
    with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
        img = src_img.read()  # 8,256,256
        for i in range(img.shape[0]):
            channel = img[i, :, :]
            channel = (channel - mean[i]) / std[i]
            img[i, :, :] = channel

        mask = src_mask.read() # 1x256x256
        mask = mask.transpose(1, 2, 0) # 256x256x1
        img = img.transpose(1, 2, 0) # 256x256x8

    return img, mask

def read_c_last(imge_path, mask_path, channel_last:True):
    with rasterio.open(imge_path) as src_img, rasterio.open(mask_path) as src_mask:
        img = src_img.read()  # 8,256,256
        mask = src_mask.read() # 1x256x256
        if channel_last:
            mask = mask.transpose(1, 2, 0) # 256x256x1
            img = img.transpose(1, 2, 0) # 256x256x8   
    return img, mask 
  

def check_errors(mask, image = None, prediction = False, pred = None):
    if prediction == False:
        h, w = image.shape[2:] # 16x8x256x256
        assert h % 32 == 0 and w % 32 == 0, f"Error! The Input Dimension is not divisible through 32 without remainder. Got {image.shape[2:]}."

        assert image.ndim == 4, f"Image has to be 4 dimensional, batch, channels, height, with but got Image Shape {image.shape}!"
        assert image.ndim == mask.ndim, f"Error Image and Mask have to be the same dimension, but got \n Image Shape: {image.shape} \n Mask Shape: {mask.shape}"
        assert image.shape[1] < image.shape[-1], f"Error! Pytorch tensor isn't channel first!, got {image.shape}" # Batch Size, Height, Width, Channel

        #assert mask.max() == float(0) and mask.min() == float(0), f"Mask should be 0 but got \n min tensor {mask.min()} \n max tensor {mask.max()}!"
    else: # prediction true
        assert pred.ndim == mask.ndim, f"Error: Prediction and mask are not the same. I got \n Predictions {pred.shape} \n Mask: {mask.shape}"

   
def get_time(end_time, start_time, object_of_interest:str):
    min = (end_time - start_time) / 60

    if min >= 60:
        hours = min // 60  # total hours
        min = min % 60   
    
    print(f"The {object_of_interest} took {hours} hours and {round(min)} minutes to run.")
       
def save_model(cfg):
    checkpoint_callback = ModelCheckpoint(
        save_top_k=20,
        monitor="val_f1",
        mode="max",
        dirpath=cfg.FOLDER.L_CHECKPOINT,
        verbose=True,
        save_last=True,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        every_n_epochs=1)
    
    return checkpoint_callback




    


        
            
            