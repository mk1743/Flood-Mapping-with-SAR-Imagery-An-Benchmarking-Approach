'''
Training of the UDA model for FO to FU domain adaptation
'''

from Dataset.Lightning_Data_Module.pl_data_module_uda import *
import lightning as Pl
from lightning.pytorch.loggers import CSVLogger
from Models.Lightning_Module.Model_UDA import ModelUDA
from Config.test.TEST_config_FU import cfg
from Utils.utils import save_model

def main(cfg):

    model = ModelUDA(cfg, test = True)

    pl_data = USFDataModule(cfg)

    logger = CSVLogger(save_dir=cfg.FOLDER.LOGGER)
                       

    checkpoint_callback = save_model(cfg)

    trainer = Pl.Trainer(accelerator=cfg.MODEL.TRAINER.ACCELERATOR ,
                         max_epochs=cfg.MODEL.TRAINER.MAX_EPOCHS ,
                         callbacks=[checkpoint_callback],
                         logger=logger,
                         deterministic=True,
                         )

    trainer.test(model = model, datamodule = pl_data, ckpt_path="/content/epoch=99-step=119200.ckpt")

    
if __name__ == "__main__":
    print(" -------------------------------------- BEGIN TEST --------------------------------------")
    main(cfg)
    
    print(" -------------------------------------- END TEST --------------------------------------")
