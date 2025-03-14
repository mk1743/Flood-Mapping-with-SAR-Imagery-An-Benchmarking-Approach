'''
Training of the UDA model for FU to FO domain adaptation
'''

from Dataset.Lightning_Data_Module.pl_data_module_uda import *
import lightning as Pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from Models.Lightning_Module.Model_UDA import ModelUDA
from Config.train.FO_config import cfg
import subprocess
import shutil
from Utils.utils import save_model

def main(cfg):

    load_model = False

    model = ModelUDA(cfg)

    pl_data = USFDataModule(cfg)

    logger = CSVLogger(save_dir=cfg.FOLDER.LOGGER,
                       name=cfg.MODEL.LOGGER.FILE_NAME)

    #stopping = EarlyStopping(monitor=cfg.MODEL.DEFAULT_METRIC,
                            # patience=cfg.EARLY_STOPPING.PATIENCE,
                            # verbose=True,
                          #  mode=cfg.MODEL.MODE,
                           #  check_on_train_epoch_end=True,
                            # min_delta=cfg.EARLY_STOPPING.MIN_DELTA,
                            # )

    checkpoint_callback = save_model(cfg)

    trainer = Pl.Trainer(accelerator=cfg.MODEL.TRAINER.ACCELERATOR ,
                         max_epochs=cfg.MODEL.TRAINER.MAX_EPOCHS,
                         callbacks=[checkpoint_callback],
                         logger=logger,
                         deterministic=True,
                         )
    
    if load_model:
        path_check = checkpoint_callback.best_model_path
        print(f"\n Path Check {path_check}")
        trainer.fit(model=model, datamodule=pl_data, ckpt_path=path_check)
    else: 
        trainer.fit(model=model, datamodule=pl_data)
    
if __name__ == "__main__":
    print(" -------------------------------------- BEGIN TRAINING --------------------------------------")
    main(cfg)
    
    print(" -------------------------------------- END TRAINING --------------------------------------")
