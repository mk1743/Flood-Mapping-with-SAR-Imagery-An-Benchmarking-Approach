from easydict import EasyDict
import os
import segmentation_models_pytorch as smp
import torch

# Test Modus
test_modus = False # default False

# HPT used yes or no

hpt_used = False 

# First instance of easydict
cfg = EasyDict()

# define what is source and what is target domain 

# Domain

## Names
cfg.DOMAIN = EasyDict()

cfg.DOMAIN.NF = "NF"
cfg.DOMAIN.FO = "FO"
cfg.DOMAIN.FU = "FU"

## Source Target Domain
cfg.DOMAIN.SOURCE = cfg.DOMAIN.FU # source domain
cfg.DOMAIN.TARGET = cfg.DOMAIN.FO # target domain 

#set name of the nodel

cfg.MODEL = EasyDict()

## Name of the models
cfg.MODEL.NAME = EasyDict()
cfg.MODEL.NAME.UDA = "UDA_Model"

# Set name of files

cfg.FILES = EasyDict()
cfg.FILES.NAME = f"{cfg.DOMAIN.SOURCE}_to_{cfg.DOMAIN.TARGET}_{cfg.MODEL.NAME.UDA}"

# Parameter HPT_USED for epoch log in model in hpt tuning

cfg.PARA = EasyDict()
cfg.PARA.HPT_USED = hpt_used 

# Folder
cfg.FOLDER = EasyDict()

cfg.FOLDER.MAIN_DIR = "/content"
cfg.FOLDER.LIGHTNING  = os.path.join(cfg.FOLDER.MAIN_DIR + "/" + cfg.FILES.NAME)
cfg.FOLDER.L_CHECKPOINT = f"{cfg.FOLDER.LIGHTNING}/CHECKPOINT"
cfg.FOLDER.LOGGER = f"{cfg.FOLDER.LIGHTNING}/LOGGER"

if hpt_used:
    cfg.FOLDER.RAYTUNE = f"{cfg.FOLDER.MAIN_DIR}/RAYTUNE"
    cfg.FOLDER.RAYTUNE_BEST = f"{cfg.FOLDER.MAIN_DIR}/RAYTUNE/BEST"
    os.makedirs(cfg.FOLDER.RAYTUNE, exist_ok=True)
    os.makedirs(cfg.FOLDER.RAYTUNE_BEST , exist_ok=True)

os.makedirs(cfg.FOLDER.LIGHTNING , exist_ok=True)
os.makedirs(cfg.FOLDER.L_CHECKPOINT, exist_ok=True)
os.makedirs(cfg.FOLDER.LOGGER, exist_ok=True)


# Reproducibility
cfg.SEED = EasyDict()
cfg.SEED.SET_SEED = 42
generator = torch.Generator()
cfg.SEED.TORCH_GENERATOR = generator.manual_seed(cfg.SEED.SET_SEED)

## Numbers
cfg.DOMAIN.NUMBER = EasyDict()
cfg.DOMAIN.NUMBER.NF = "01"
cfg.DOMAIN.NUMBER.FO = "02"
cfg.DOMAIN.NUMBER.FU = "03"

# Paths (EXP = experiments)
cfg.PATHS = EasyDict()

## Data Paths

cfg.PATHS.DATAPATHS = EasyDict()

cfg.PATHS.DATAPATHS.CSV_SOURCE = f"/content/{cfg.DOMAIN.SOURCE}_256.csv"
cfg.PATHS.DATAPATHS.CSV_TARGET = f"/content/{cfg.DOMAIN.TARGET}_256.csv"
#cfg.PATHS.DATAPATHS.CSV_NF = f"/content/{cfg.DOMAIN.NF}_256.csv"

cfg.PATHS.DATAPATHS.CSV_TEST = None

# Preprocessing
cfg.PREPROCESSING = EasyDict()
cfg.PREPROCESSING.PATCH_SIZE = 256
cfg.PREPROCESSING.DROP_RATE = 0.40

# Data

# Train Test SPlit
cfg.DATA = EasyDict()
cfg.DATA.SPLIT = 0.20

## weight factors for parameter alpha in focal loss
cfg.DATA.WEIGHTS_FO = 3.377033232438419
cfg.DATA.WEIGHTS_FU = 20.175890320787637

# Lightning Data Module
## Weighted Random Sampling

cfg.DATA.WEIGHTS_PATH = EasyDict()
#cfg.DATA.WEIGHTS_PATH.NF = f"{cfg.FOLDER.MAIN_DIR}/weights_{cfg.DOMAIN.NUMBER.NF}_{cfg.DOMAIN.NF}.pkl"
cfg.DATA.WEIGHTS_PATH.FO = f"{cfg.FOLDER.MAIN_DIR}/weights_{cfg.DOMAIN.NUMBER.FO}_{cfg.DOMAIN.FO}.pkl"
cfg.DATA.WEIGHTS_PATH.FU = f"{cfg.FOLDER.MAIN_DIR}/weights_{cfg.DOMAIN.NUMBER.FU}_{cfg.DOMAIN.FU}.pkl"

cfg.DATA.WEIGHTS_PATH.SOURCE_DOMAIN = cfg.DATA.WEIGHTS_PATH.FO if cfg.DOMAIN.SOURCE\
else cfg.DATA.WEIGHTS_PATH.FU
cfg.DATA.WEIGHTS_PATH.TARGET_DOMAIN = cfg.DATA.WEIGHTS_PATH.FO if cfg.DOMAIN.TARGET\
else cfg.DATA.WEIGHTS_PATH.FU  

## DataLoader
cfg.DATA.BATCH_SIZE = 8 
cfg.DATA.PIN_MEMORY = True
cfg.DATA.NUM_WORKERS = os.cpu_count()
cfg.DATA.TORCH = "cuda"
cfg.DATA.PERSISTENT_WORKERS = True

# Model

## Normalization
cfg.MODEL.MEAN = [0.23651549,0.31761484,0.18514981,0.26901252, -14.57879175,  -8.6098158,  -14.29073382,-8.33534564]
cfg.MODEL.STD = [0.16280619, 0.20849304, 0.14008107, 0.19767644, 4.07141682, 3.94773216, 4.21006244, 4.05494136]

## Parameter smp
cfg.MODEL.ARCH  = "UnetPlusPlus"
cfg.MODEL.DECODER = "efficientnet-b7"
cfg.MODEL.WEIGHTS = None
cfg.MODEL.IN_CHANNELS = 8
cfg.MODEL.OUT_CHANNELS = 1

## Optimizer
cfg.MODEL.OPTIMIZER = EasyDict()
'''
cfg.MODEL.OPTIMIZER.ADAMW.LR = 0.0002667538336 if domain_nf_used else None
cfg.MODEL.OPTIMIZER.ADAMW.WEIGHT_DECAY = 4.52190373e-05 if domain_nf_used else None 
cfg.MODEL.OPTIMIZER.ADAMW.BETA_1 = 0.9612936266106 if domain_nf_used else None
cfg.MODEL.OPTIMIZER.ADAMW.BETA_2 = 0.9420615116482 if domain_nf_used else None

'''
cfg.MODEL.OPTIMIZER.NAME = "AdamW"

cfg.MODEL.OPTIMIZER.ADAMW = EasyDict()
cfg.MODEL.OPTIMIZER.ADAMW.LR = 1.09596045e-05
cfg.MODEL.OPTIMIZER.ADAMW.WEIGHT_DECAY = 4.9281223e-05
cfg.MODEL.OPTIMIZER.ADAMW.BETA_1 = 0.978325014713
cfg.MODEL.OPTIMIZER.ADAMW.BETA_2 = 0.8014125544134

## LOSS
cfg.MODEL.LOSS = EasyDict()

cfg.MODEL.LOSS.ALPHA = cfg.DATA.WEIGHTS_FO if cfg.DOMAIN.SOURCE == "FO" else cfg.DATA.WEIGHTS_FU # only for supervised loss

cfg.MODEL.LOSS.GAMMA = 4.8687775942073
'''
Loss only for the source domain (supervised loss)
'''
cfg.MODEL.LOSS.LOSS = smp.losses.FocalLoss(mode = "binary", 
                                           gamma=cfg.MODEL.LOSS.GAMMA,
                                           alpha = cfg.MODEL.LOSS.ALPHA) 

cfg.MODEL.LOSS.VAL_LOSS = smp.losses.FocalLoss(mode = "binary")

## Metric
cfg.MODEL.DEFAULT_METRIC = "val_loss" if hpt_used else "val_f1"
cfg.MODEL.MODE = "min" if hpt_used else "max"

# CSV Logger
cfg.MODEL.LOGGER = EasyDict()

cfg.MODEL.LOGGER.FILE_NAME = f"logger_{cfg.FILES.NAME}"

## Lightning Trainer
cfg.MODEL.TRAINER = EasyDict()
cfg.MODEL.TRAINER.DEVICE = -1 # use all available divces for training
cfg.MODEL.TRAINER.ACCELERATOR = "gpu"
cfg.MODEL.TRAINER.PROGRESS_BAR = False if hpt_used else True 
cfg.MODEL.TRAINER.MAX_EPOCHS = 1 if test_modus else 100 # infinity training (early stopping or raytune will be terminate training)

## Lightning Checkpoint
cfg.MODEL.CHECKPOINT = EasyDict()
cfg.MODEL.CHECKPOINT.TOP_K = 20
cfg.MODEL.CHECKPOINT.SAVE_LAST = True
cfg.MODEL.CHECKPOINT.AUTO_INSERT = True
cfg.MODEL.CHECKPOINT.ON_TRAIN_EPOCH = False
cfg.MODEL.CHECKPOINT.EVERY_N = 1
cfg.MODEL.CHECKPOINT.VERBOSE  = True

### Checkpoint Name
cfg.MODEL.CHECKPOINT.FILENAME = EasyDict()
cfg.MODEL.CHECKPOINT.FILENAME = os.path.join(cfg.FOLDER.L_CHECKPOINT + "/" + f"checkpoint_{cfg.FILES.NAME}")

# Early Stopping
cfg.EARLY_STOPPING = EasyDict()
cfg.EARLY_STOPPING.PATIENCE = 10 # if there is no improvement after 10 epochs, the training will be stopped
cfg.EARLY_STOPPING.MIN_DELTA = 0.00005
# Scheduler
cfg.LRSCHEDULER =EasyDict()

cfg.LRSCHEDULER.ACTIVATE = False if hpt_used else True 
cfg.LRSCHEDULER.FACTOR = 0.50
cfg.LRSCHEDULER.PATIENCE  = 3

if hpt_used:

    # torchmetrics
    cfg.METRICS = EasyDict()
    cfg.METRICS.PROG_BAR = False

    # HPT
    cfg.TUNING = EasyDict()

    ## Checkpoint
    cfg.TUNING.CHECKPOINT = EasyDict()
    cfg.TUNING.CHECKPOINT.FILENAME = os.path.join(cfg.FOLDER.RAYTUNE + "/" +  f"raytune_{cfg.FILES.NAME}") 

    ## HyperBandForBOHB
    cfg.HBFB = EasyDict()
    cfg.HBFB.REDUCTION_FACTOR = 3
    cfg.HBFB.MAX_T = 9

    ## Tuner
    cfg.TUNE_CONFIG = EasyDict()

    cfg.TUNE_CONFIG.RESSOURCES = {"cpu": 12, "gpu": 1} 
    cfg.TUNE_CONFIG.WORKERS = 1
    cfg.TUNE_CONFIG.NUM_SAMPLES = 5
    cfg.RUN_CONFIG = EasyDict()

    cfg.RUN_CONFIG.NAME = f"HPT_{cfg.FILES.NAME}" 
    cfg.RUN_CONFIG.NAME_PATH = cfg.FOLDER.RAYTUNE

# Torchmetrics
cfg.METRICS = EasyDict()
cfg.METRICS.PROG_BAR = True

# OT
cfg.OT = EasyDict()
cfg.OT.LAMBDA_STEP = 0.9832308858068
cfg.OT.ALPHA = 1.5
cfg.OT.BETA  = 2.5


# Visualization
cfg.LOGER = EasyDict()
cfg.LOGER.FILENAME = EasyDict()
cfg.LOGER.FILENAME = f"logs_{cfg.FILES.NAME}"