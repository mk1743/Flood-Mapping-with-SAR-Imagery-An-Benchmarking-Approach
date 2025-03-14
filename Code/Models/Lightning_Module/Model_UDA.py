import lightning as pl
import segmentation_models_pytorch as smp
from UDA.OT.ot import *
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import JaccardIndex
from torchvision.transforms import Normalize

from Utils.utils import check_errors
import time
'''
For segmentation_models_pytorch
https://github.com/qubvel-org/segmentation_models.pytorch
'''

'''
Source: 

https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb#:~:text=Binary%20segmentation%20intro%20-%20Colab.%20%F0%9F%87%AD

'''

class ModelUDA(pl.LightningModule):
    def __init__(self, cfg, test):
        super().__init__()
        
        self.cfg = cfg
        # Pytorch Segmentation Model
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=self.cfg.MODEL.WEIGHTS,
            in_channels=self.cfg.MODEL.IN_CHANNELS ,
            classes=self.cfg.MODEL.OUT_CHANNELS,)
        
        # Loss function
        if not test:
            self.loss_fn = self.cfg.MODEL.LOSS.LOSS # supervised loss for training

        # alignment instances for OT
        self.f_alignment = FeatureAlignment(alpha = self.cfg.OT.ALPHA ) # alpha default value 1
        self.l_alignment = LabeAlignment(beta = self.cfg.OT.BETA) # beta default value 1

        # Metric instanciation
        
        # validation
        self.f1 = F1Score(task="binary", num_classes=1,threshold=0.50)
        self.precision = Precision(task="binary", num_classes=1,threshold=0.50)
        self.recall = Recall(task="binary", num_classes=1,threshold=0.50)
        self.iou_0 = JaccardIndex(task="binary", num_classes=1,threshold=0.50)
        self.iou_1 = JaccardIndex(task="binary", num_classes=1,threshold=0.50)

        # Test
        self.f1_test = F1Score(task="binary", num_classes=1,threshold=0.50, ignore_index=-1)
        self.precision_test = Precision(task="binary", num_classes=1,threshold=0.50, ignore_index=-1)
        self.recall_test = Recall(task="binary", num_classes=1,threshold=0.50, ignore_index=-1)
        self.iou_0_test = JaccardIndex(task="binary", num_classes=1,threshold=0.50, ignore_index=-1)
        self.iou_1_test = JaccardIndex(task="binary", num_classes=1,threshold=0.50, ignore_index=-1)
        
        #train 
        #self.f1_train = F1Score(task="binary", num_classes=1,threshold=0.50)
        #self.precision_train = Precision(task="binary", num_classes=1,threshold=0.50)
        #self.recall_train = Recall(task="binary", num_classes=1,threshold=0.50)
        #self.iou_0_train = JaccardIndex(task="binary", num_classes=1,threshold=0.50)
        #self.iou_1_train = JaccardIndex(task="binary", num_classes=1,threshold=0.50)


        # Normalisation
        self.norm = Normalize(mean = self.cfg.MODEL.MEAN , std = self.cfg.MODEL.STD)

    def forward(self, img): # method for both domains source and target; img already band like normalized
        # Normalize
        img = self.norm.forward(img)
        # forward pass with backbone and UNet model
        backbone_features = self.model.encoder(img) # features from feature extractor (backbone)
        backbone_features = backbone_features[-1] # only use the features from the last layer of the backbone model 
        model_features = self.model(img) # features from the whole model inclusive backbone
        
        return backbone_features, model_features

    def training_step(self, batch, batch_idx): # train loop one bacht
                                                           
        # get input Data from nested dictionary
        source_img = batch['source_loader']["image"] # training image for the source domain
        source_mask = batch['source_loader']["mask"] # training mask for the source domain


        target_img = batch['target_loader']["image"] # target domain
        target_mask = batch['target_loader']["mask"] # for error check only

        # check for errors source domain

        check_errors(image= source_img, mask = source_mask)
        check_errors(image = target_img, mask =  target_mask)

        assert source_img.shape == target_img.shape,\
        f"Erorr Source IMG and Target IMG are not the same got Source IMG {source_img.shape} and Target IMG {target_img.shape}"

        # UDA-OT
        
        # forward pass
        backbone_features_source, model_features_source = self.forward(source_img) # forward pass for source domain
        backbone_features_target, model_features_target = self.forward(target_img) # forward pass for target domain

        # Check prediction for errors
        check_errors(prediction=True, pred=backbone_features_source, mask=source_img)
        check_errors(prediction=True, pred=model_features_source, mask=source_img)

        check_errors(prediction=True, pred=backbone_features_target, mask=target_mask)
        check_errors(prediction=True, pred=model_features_target, mask=target_mask)

        # feature alignment for source and target features
        feat_loss, sorting = self.f_alignment.compute(backbone_features_source, backbone_features_target) # predictions from backbone model from source and target domain; (8) 

        # label alignment
        ground_truth = order(source_mask, sorting) # Use of source masks, as there are no masks to train in the target domain; list with tensors [tensor1, tensor2,...,tensor16]
        ground_truth = torch.stack(ground_truth) # torch.tensor (16,1,256,256)
        label_pred = model_features_target

        label_loss = self.l_alignment.compute(ground_truth, label_pred) # ground_truth --> sorted from source domain; (8)

        # supervised loss for the model (focal loss)
        sup_loss = self.loss_fn(y_pred = model_features_source, # Focal Loss
                            y_true = source_mask)

        # total loss calculation
        #lambda_step = 0.01  # from paper
        total_loss = sup_loss + self.cfg.OT.LAMBDA_STEP * (feat_loss + label_loss) # (7) from paper

        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True, logger=True)
        
        # learning rate loggen
        opt = self.optimizers() # optimizer
        lr = opt.param_groups[0]['lr'] # Takes the LR from the first parameter group
        self.log('lr', lr, prog_bar=False, logger=True, on_epoch=True)

        # Metrics

        # Metric calculation
        #self.iou_0_train(model_features_target, target_mask)
        #self.iou_1_train(model_features_target, target_mask)

        #f1 = self.f1_train(model_features_target, target_mask)
        #precision = self.precision_train(model_features_target, target_mask)
        #recall = self.recall_train(model_features_target, target_mask)

        #self.log_dict({"train_f1":f1, "train_precision":precision, "train_recall": recall}, prog_bar=True, on_epoch=True, logger=True)

        return total_loss
    
    #def on_train_epoch_end(self):
     #   iou_0 = self.iou_0_train.compute() # torch.tensor with no dimension torch.tensor(number)
      #  iou_1= self.iou_1_train.compute() # torch.tensor with no dimension torch.tensor(number)

      #  miou = (iou_0 + iou_1)/2.

       # self.log("train_miu", miou, prog_bar=True, logger=True)

#        self.iou_0_train.reset()
 #       self.iou_1_train.reset()
    
    def validation_step(self, batch, batch_idx):
        img = batch["image"] # valdiation Data from target domain
        mask = batch["mask"]

        # Check for errors
        check_errors(image = img, mask = mask)

        _, output = self.forward(img) # torch.Size([16, 1, 256, 256])

        # Check for errors in prediction
        check_errors(mask = mask, prediction=True, pred = output)

        # Dice loss calculation 
        loss = self.loss_fn(y_pred=output,y_true=mask) # logits values 

        # Metric calculation
        self.iou_0(output, mask)
        self.iou_1(output, mask)

        f1 = self.f1(output, mask)
        precision = self.precision(output, mask)
        recall = self.recall(output, mask)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        self.log_dict({"val_f1":f1, "val_precision":precision, "val_recall": recall}, prog_bar=True, on_epoch=True, logger=True)
        
        if self.cfg.PARA.HPT_USED:
        
            self.log("epoch", self.current_epoch, prog_bar = False, on_epoch = True)

        return loss
    
    def on_validation_epoch_end(self):
        
        iou_0 = self.iou_0.compute() # torch.tensor with no dimension torch.tensor(number)
        iou_1= self.iou_1.compute() # torch.tensor with no dimension torch.tensor(number)

        miou = (iou_0 + iou_1)/2.

        self.log("val_mioU", miou, prog_bar=True, logger=True)

        self.iou_0.reset()
        self.iou_1.reset()

    def test_step(self, batch, batch_idx):
        img = batch["image"] # valdiation Data from target domain
        mask = batch["mask"]

        # Check for errors
        check_errors(image = img, mask = mask)

        _, output = self.forward(img) # torch.Size([16, 1, 256, 256])

        # Check for errors in prediction
        check_errors(mask = mask, prediction=True, pred = output)

        # Metric calculation
        self.iou_0_test(output, mask)
        self.iou_1_test(output, mask)

        f1 = self.f1_test(output, mask)
        precision = self.precision_test(output, mask)
        recall = self.recall_test(output, mask)

        self.log_dict({"test_f1":f1, "test_preicsion":precision, "test_recall": recall}, prog_bar=True, on_epoch=True, logger=True)
        
        if self.cfg.PARA.HPT_USED:
        
            self.log("epoch", self.current_epoch, prog_bar = False, on_epoch = True)

    def on_test_epoch_end(self):
        
        iou_0 = self.iou_0_test.compute() # torch.tensor with no dimension torch.tensor(number)
        iou_1= self.iou_1_test.compute() # torch.tensor with no dimension torch.tensor(number)

        miou = (iou_0 + iou_1)/2.

        self.log("test_mioU", miou, prog_bar=True, logger=True)

        self.iou_0_test.reset()
        self.iou_1_test.reset()
    
    def configure_optimizers(self): 
        adamw = torch.optim.AdamW(self.parameters(),
                                  lr = self.cfg.MODEL.OPTIMIZER.ADAMW.LR,
                                  weight_decay=self.cfg.MODEL.OPTIMIZER.ADAMW.WEIGHT_DECAY,
                                  betas=(self.cfg.MODEL.OPTIMIZER.ADAMW.BETA_1,self.cfg.MODEL.OPTIMIZER.ADAMW.BETA_2))

        if self.cfg.LRSCHEDULER.ACTIVATE:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adamw, 
                                                            mode='max', 
                                                            factor=self.cfg.LRSCHEDULER.FACTOR,
                                                            patience=self.cfg.LRSCHEDULER.PATIENCE)
            return {
                'optimizer': adamw,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': self.cfg.MODEL.DEFAULT_METRIC,
                    'interval': 'epoch',
                    'frequency': 1
                }}
        else:
            return adamw

