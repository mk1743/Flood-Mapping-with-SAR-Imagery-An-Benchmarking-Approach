
from Dataset.dataset_uda import UrbanSARFloods
import lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from Utils.utils_train import sampler
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader

# Code based on: https://gist.github.com/ashleve/ac511f08c0d29e74566900fd3efbb3ec


class USFDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        #self.source = pd.read_csv(self.cfg.PATHS.DATAPATHS.CSV_SOURCE, header = 0, sep = ",") # pd.DataFrame Source Domain
        #self.target = pd.read_csv(self.cfg.PATHS.DATAPATHS.CSV_TARGET, header = 0, sep = ",") # pd.DataFrame Target Domain
        self.test_data = pd.read_csv("/content/Test_256.csv", header = 0, sep = ",") 

        self.path_class_weights_source = self.cfg.DATA.WEIGHTS_PATH.SOURCE_DOMAIN
        self.path_class_weights_target = self.cfg.DATA.WEIGHTS_PATH.TARGET_DOMAIN

    def setup(self, stage:str):

        if stage == "fit": # traning
            # generate weights for WeightedRandomSampler from utils_train
            self.weights_source = sampler(self.source, self.path_class_weights_source) # list; para df, class_weights_path
            self.weights_target = sampler(self.target, self.path_class_weights_target) # list 

            assert len(self.source) == len(self.weights_source), \
            f"Length of weights {len(self.weights_source)} and length of data frame {len(self.source)} are not the same!"
            
            assert len(self.target) == len(self.weights_target), \
            f"Length of weights {len(self.weights_target)} and length of data frame {len(self.target)} are not the same!"

            # Dataset
            ## Source Domain 
            
            # convert weight to tensor
            self.train_source_weights = torch.from_numpy(self.weights_source) # float 64

            ### Dataset train

            self.dataset_source_train = UrbanSARFloods(df=self.source,split="train")
            
            ## Target Domain

            ### Train Test Split
            train_idx_target,val_idx_target, w_train_target, _ =train_test_split(self.target, 
                                                                            self.weights_target, 
                                                                            test_size=self.cfg.DATA.SPLIT, 
                                                                            random_state=self.cfg.SEED.SET_SEED, 
                                                                            shuffle=True)
            
            ### Dataset train
            self.dataset_target_train = UrbanSARFloods(df=train_idx_target,split="train")

            # Dataset val
            self.dataset_target_val = UrbanSARFloods(df = val_idx_target,split = "val")
            
            # convert weight to tensor
            self.train_target_weights = torch.from_numpy(w_train_target) # float 64
  
            # print length dataset for source and target domain 
            print(f"Length Dataset Source Train {self.dataset_source_train}")
            print(f"Length Dataset Target Train {self.dataset_target_train}")
        
        else: # test
            self.test_dataset = UrbanSARFloods(df=self.test_data,split="test")

            print(f"Length Dataset Test {self.test_dataset}")
            
    # Data loader for training (source and target domain) 
    def train_dataloader(self):

        # Source domain Sampler
        sampler_source = WeightedRandomSampler(weights = self.train_source_weights, 
                                               num_samples=(len(self.train_source_weights)), 
                                               replacement=True,
                                               generator=self.cfg.SEED.TORCH_GENERATOR)
        
        #Source Domain DataLoader
        
        source_train_loader = DataLoader(dataset= self.dataset_source_train,
                                        batch_size=self.cfg.DATA.BATCH_SIZE,
                                        pin_memory=self.cfg.DATA.PIN_MEMORY,
                                        num_workers=self.cfg.DATA.NUM_WORKERS,
                                        pin_memory_device=self.cfg.DATA.TORCH,
                                        persistent_workers=self.cfg.DATA.PERSISTENT_WORKERS,
                                        drop_last=True,
                                        sampler = sampler_source)
        

        # Target domain data laoder
        sampler_train_target = WeightedRandomSampler(weights = self.train_target_weights, 
                                                    num_samples=(len(self.train_target_weights)), 
                                                    replacement=True,
                                                    generator=self.cfg.SEED.TORCH_GENERATOR)
        

        target_train_loader = DataLoader(dataset=self.dataset_target_train,
                                        batch_size=self.cfg.DATA.BATCH_SIZE,
                                        pin_memory=self.cfg.DATA.PIN_MEMORY,
                                        num_workers=self.cfg.DATA.NUM_WORKERS,
                                        pin_memory_device=self.cfg.DATA.TORCH,
                                        persistent_workers=self.cfg.DATA.PERSISTENT_WORKERS,
                                        drop_last=True,
                                        sampler=sampler_train_target)
        
        iterables = {"source_loader":source_train_loader, "target_loader":target_train_loader}

        return  CombinedLoader(iterables)

    def val_dataloader(self):  
        # For Target Domain
        target_val_loader = DataLoader(dataset=self.dataset_target_val,
                                     batch_size=self.cfg.DATA.BATCH_SIZE,
                                     pin_memory=self.cfg.DATA.PIN_MEMORY,
                                     num_workers=self.cfg.DATA.NUM_WORKERS,
                                     pin_memory_device=self.cfg.DATA.TORCH,
                                     persistent_workers=self.cfg.DATA.PERSISTENT_WORKERS,
                                     drop_last=True,
                                     shuffle= False)
        
        return target_val_loader
       
    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset,
                                  batch_size=self.cfg.DATA.BATCH_SIZE,
                                  pin_memory=self.cfg.DATA.PIN_MEMORY,
                                  num_workers=self.cfg.DATA.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory_device=self.cfg.DATA.TORCH,
                                  persistent_workers=self.cfg.DATA.PERSISTENT_WORKERS,
                                  drop_last=True)                               
        
        return test_loader