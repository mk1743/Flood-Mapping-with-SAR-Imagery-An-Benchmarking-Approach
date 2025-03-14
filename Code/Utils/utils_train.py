import os.path as osp
import numpy as np
import os
from sklearn.model_selection import KFold
import torch
import rasterio
import torch 
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

'''
Adapted from https://github.com/ZJunBo/agricultural-land-extraction/blob/79581140549b65e46e384d7623617a5e155a85d1/agricultural-land-extraction-master/ADVENT/advent/scripts/train.py
(retrieved on 2025-01-20)
'''

def generate_paths(cfg):
    #Snapshot Pahs
    if cfg.Paths.EXP.EXP_Name == '':
        cfg.Paths.EXP.EXP_Name = f"{cfg.Paths.SOURCE}_to_{cfg.Paths.TARGET}_{cfg.Paths.EXP.MODEL_NAME}_{cfg.Paths.EXP.UDA}"

    if cfg.Paths.SNAPSHOT_DIR == '':
        cfg.Paths.SNAPSHOT_DIR = osp.join(cfg.Paths.EXP.EXP_ROOT_SNAPSHOT, cfg.Paths.EXP.EXP_Name)
        os.makedirs(cfg.Paths.SNAPSHOT_DIR, exist_ok=True)

    # Tensorboard Paths
    if cfg.Paths.TENSORBOARD.TENSORBOARD_LOGDIR == "":
        cfg.Paths.TENSORBOARD.TENSORBOARD_LOGDIR = osp.join(cfg.Paths.EXP.EXP_ROOT_LOGS, 'tensorboard', cfg.Paths.EXP.EXP_Name)
        os.makedirs(cfg.Paths.TENSORBOARD.TENSORBOARD_LOGDIR, exist_ok=True)


def generate_folds_uda(df_source, df_target, cfg):
    fold_data_list = []
    kf_source = KFold(n_splits=cfg.DATA.kfold, shuffle=True, random_state=cfg.DATA.random_state)
    kf_target = KFold(n_splits=cfg.DATA.kfold, shuffle=True, random_state=cfg.DATA.random_state)

    for (source_train_index, _), (target_train_index, target_val_index) in zip(kf_source.split(df_source),kf_target.split(df_target)):

        fold_data = {
            "source_train_index": df_source.iloc[source_train_index],
            "target_train_index": df_target.iloc[target_train_index],
            "target_val_index": df_target.iloc[target_val_index]
        }
        fold_data_list.append(fold_data)

    return fold_data_list

def generate_folds(df,cfg):

    kf_fold = KFold(n_splits=cfg.DATA.kfold, shuffle=True, random_state=cfg.DATA.random_state)

    fold_data_list = []

    for train_index, test_index in kf_fold.split(df): # df = pd.DataFrame
        fold_data = {
        "train_index":df.iloc[train_index],
        "val_index":df.iloc[test_index]
        }
        fold_data_list.append(fold_data)

    return fold_data_list

def read_tif(img_path, mask_path):
    with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
        img = src_img.read() # 8,256,256
        img = img.transpose(1,2,0)
        
        mask = src_mask.read() # 1,256,256
        mask = mask.transpose(1,2,0)
        
        return img, mask # channel last
        
def logits_to_probabilities(input:torch.tensor):
    prob = input.sigmoid()
    prob = (prob >=0.50).int()

    return prob

def give_class_weights(path_to_df:pd.DataFrame, classes:list):
    df = pd.read_csv(path_to_df, header = 0, sep = ",")
    pixel_per_class = {0:0, 1:0,}
    total_class_pixel = {0:0, 1:0}
    prob = []
    weights = {0:0., 1:0.}

    for i in range(len(df)):
        mask_path = df.iloc[i]["Mask_Paths"]
        mask = rasterio.open(mask_path).read() # 1x256x256 float32 numpy 

        for c in classes:
            if np.any(mask == float(c)):
                class_pixel = 256*256
                total_class_pixel[c] += class_pixel # Total number of pixels of all images in which class c occurs
        
        unqiue_values, counts = np.unique(mask, return_counts=True)
        for unqiue_value, count in zip(unqiue_values, counts): # numpy value <class 'numpy.float32'>
            unqiue_value = int(unqiue_value)
            count = int(count)
            pixel_per_class[unqiue_value] += count

    for (_, value_per_class), (_, value_total_pixel) in zip(sorted(pixel_per_class.items()), sorted(total_class_pixel.items())): # sorting by key values
        prob.append(value_per_class/value_total_pixel)
       
    median = np.median(prob)

    for (key, prob_value) in zip(sorted(weights.keys()), prob):
        weights[key] = median/prob_value
                  
    return weights
    
def sampler(df, class_weights_path):
    weights_per_image = []

    print(f"\n Beginn calculate weights per image!")

    for i in range(len(df)):
        mask_path = df.iloc[i]["Mask_Paths"]
        mask = rasterio.open(mask_path).read()

        with open(class_weights_path, 'rb') as weights_pkl:
        
            unique_elements, counts = np.unique(mask, return_counts=True)

            pixel_per_image_dic = dict(zip(unique_elements, counts)) # e.g. dic {0:1000, 1:10000, -1:50000}

            weights = pickle.load(weights_pkl)

            weighted_sum = 0 # Resetting the weighted sum per image

            for class_label, pixel_count in sorted(pixel_per_image_dic.items()): # loop runs two times for two keys 0,1
                weighted_sum += pixel_count * weights.get(class_label, 0)  # 0 als default value if class label not in weights 

            weights_per_image.append(weighted_sum)
    
    print(f"\n End of the weight calculation")

    return np.array(weights_per_image) # return of a list with weights per domain 