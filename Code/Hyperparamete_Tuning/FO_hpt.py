'''
Hyperparameter tuning for domain adaptation FU to FO. 
'''

# Imports Raytune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import tune
from ray.tune.tuner import Tuner
from ray.tune import TuneConfig
from ray.train import RunConfig
from ray.tune import with_resources
import ray 

# Lightning
import lightning as pl
from Models.Lightning_Module.Model_UDA import *
from Dataset.Lightning_Data_Module.pl_data_module_uda import * 

# Config
from Config.train.FO_config import cfg # config for FU to FO (source -- target)

# miscellaneous
from copy import deepcopy
import time
from Utils.utils import get_time
import numpy as np 
import os

def config_space():

    config_space = CS.ConfigurationSpace()

    # 1. Hyperparameters for optimizer
    lr_adam = CSH.UniformFloatHyperparameter("lr_adam", lower=10 ** -5, upper=10 ** -2, log=True,)
    wd_adam = CSH.UniformFloatHyperparameter("wd_adam", lower=1e-5, upper = 1e-2, log=True,)
    beta1_adam = CS.UniformFloatHyperparameter(name="beta1_adam", lower=0.8, upper=0.99,log=False)
    beta2_adam = CS.UniformFloatHyperparameter(name="beta2_adam", lower=0.8, upper=0.9999, log = False)

    config_space.add(lr_adam)
    config_space.add(wd_adam)
    config_space.add(beta1_adam)
    config_space.add(beta2_adam)

    # 2. Hyperparameter for loss function
    gamma_loss = CSH.UniformFloatHyperparameter("gamma", lower=0, upper=5, log=False)

    config_space.add(gamma_loss)

    # 3. Hyperparameter for OT Method (UDA)
    lambda_ot = CSH.UniformFloatHyperparameter("lambda", lower=0, upper=1, log=False)
    alpha_ot = CSH.CategoricalHyperparameter("alpha", choices=[1,1.5,2,2.5,3.]) # max 4 values with num_samples = 4
    beta_ot = CSH.CategoricalHyperparameter("beta", choices=[1,1.5,2,2.5,3.]) # max 4 values with num_samples = 4

    config_space.add(lambda_ot)
    config_space.add(alpha_ot)
    config_space.add(beta_ot)

    # 5. Additional: Batch Size
    batch_size = CSH.CategoricalHyperparameter("Batch_Size", choices=[2,4,6,8,10]) #max 4 values with num_samples = 4

    config_space.add(batch_size)


    return config_space

def train_function(config):
    '''
    ray tune gives the values as numpy array back but pytorch expect an integer value for batch size!
    '''
    trial_config = deepcopy(cfg) # cfg own USFDataModuleconfiguration file not equal to train_function(config) config --> config from raytune --> config_space

    # Set hyperparameters to config file
    ## Optimizer
    trial_config.MODEL.OPTIMIZER.ADAMW.LR = float(config["lr_adam"]) # ray tune gives back a numyp ndarray, but torch expected python float values 
    trial_config.MODEL.OPTIMIZER.ADAMW.WEIGHT_DECAY = float(config["wd_adam"])
    trial_config.MODEL.OPTIMIZER.ADAMW.BETA_1 = float(config["beta1_adam"])
    trial_config.MODEL.OPTIMIZER.ADAMW.BETA_2 = float(config["beta2_adam"])

    ## loss
    trial_config.MODEL.LOSS.GAMMA = float(config["gamma"])

    ## OT Method (UDA)
    trial_config.OT.LAMBDA_STEP = float(config["lambda"])
    trial_config.OT.ALPHA = float(config["alpha"])
    trial_config.OT.BETA = float(config["beta"])

    ## Batch Size
    trial_config.DATA.BATCH_SIZE = int(config["Batch_Size"])

    # set model and lightning data module
    pl_module = USFDataModule(trial_config)
    model = ModelUDA(trial_config)

    tune_callback = TuneReportCheckpointCallback(
        metrics={"val_loss":"val_loss", "epoch":"epoch"},
        save_checkpoints = False,
        on="validation_epoch_end")

    trainer = pl.Trainer(
        devices=trial_config.MODEL.TRAINER.DEVICE,
        accelerator=trial_config.MODEL.TRAINER.ACCELERATOR,
        callbacks=[tune_callback],
        enable_progress_bar=trial_config.MODEL.TRAINER.PROGRESS_BAR,
        max_epochs=trial_config.MODEL.TRAINER.MAX_EPOCHS
    )

    trainer.fit(model = model, datamodule=pl_module)

def optimizer(cfg):
    
    # search algorithmen
    search_alg = TuneBOHB(space=config_space(),seed=cfg.SEED.SET_SEED, metric = cfg.MODEL.DEFAULT_METRIC,mode = cfg.MODEL.MODE)

    # Scheduler (early stopping)
    scheduler = HyperBandForBOHB(time_attr="epoch",max_t=cfg.HBFB.MAX_T, reduction_factor=cfg.HBFB.REDUCTION_FACTOR)

    return search_alg , scheduler

def run(cfg):

    ray.init(runtime_env = {"env_vars": {"PYTHONPATH": "/content/Code"}})

    ressources = with_resources(trainable= train_function, resources=cfg.TUNE_CONFIG.RESSOURCES)

    search_alg , scheduler = optimizer(cfg)

    tune_config = TuneConfig(mode = cfg.MODEL.MODE , 
                             metric=cfg.MODEL.DEFAULT_METRIC, 
                             search_alg = search_alg, 
                             scheduler=scheduler, 
                             num_samples=cfg.TUNE_CONFIG.NUM_SAMPLES,
                             max_concurrent_trials = cfg.TUNE_CONFIG.WORKERS,
                             reuse_actors = True,
                             time_budget_s = 5*60*60
                            )
    
    run_config = RunConfig(name=cfg.RUN_CONFIG.NAME, storage_path=cfg.RUN_CONFIG.NAME_PATH)

    tuner = Tuner(trainable = ressources,
                  tune_config=tune_config,
                  run_config=run_config,
                  )
    
    tuner.fit()
    
if __name__ == "__main__":
    start_time = time.time()
    run(cfg)
    end_time = time.time()
    get_time(end_time, start_time, "Hyperparameter Tuning")

    analysis = tune.ExperimentAnalysis(os.path.join(cfg.RUN_CONFIG.NAME_PATH, cfg.RUN_CONFIG.NAME))
    best_config = analysis.get_best_config(metric=cfg.MODEL.DEFAULT_METRIC, mode=cfg.MODEL.MODE)
    print("Best Hyperparameters:", best_config)
   




    
    
    
    
    
    