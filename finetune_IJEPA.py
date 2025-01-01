import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from model import IJEPA_base
from pretrain_IJEPA import IJEPA

from datamodules import ZipDataModule

'''
Finetune IJEPA
'''
class IJEPA_FT(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1):

        super().__init__()
        self.save_hyperparameters()

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        if pretrained_model_path is None: # if no pretrained model path is given, create a new model
            self.pretrained_model = IJEPA(img_size=224, patch_size=16, in_chans=3, embed_dim=64, enc_heads=8, enc_depth=8, decoder_depth=6, lr=1e-3)
        else:
            self.pretrained_model = IJEPA.load_from_checkpoint(pretrained_model_path)
        self.pretrained_model.model.mode = "test"
        self.pretrained_model.model.layer_dropout = self.drop_path
        self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.num_tokens),
            nn.Linear(self.pretrained_model.num_tokens, num_classes),
        )

        #define loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pretrained_model.model(x)
        x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x) #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y) #calculate loss
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean() #calculate accuracy
        self.log('train_accuracy', accuracy)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch[1])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

# python3 finetune_IJEPA.py
if __name__ == '__main__':
    import pandas as pd
    zip_df = pd.read_csv('supervised_zipfiles.csv')
    datamodule = ZipDataModule(dataset_df=zip_df, num_classes=2)

    experiment_name = 'early_pretrained'

    #model = IJEPA_FT(pretrained_model_path=None, num_classes=2)
    model = IJEPA_FT(pretrained_model_path='pretrain/pretrain-epoch=00-val_loss=0.02.ckpt', num_classes=2)

    tensorboard_logger = pl.loggers.TensorBoardLogger('tb_logs/finetune/', default_hp_metric=False, name=experiment_name)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath=os.path.join('finetune',experiment_name), filename='finetune-{epoch:02d}-{val_loss:.2f}')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    #model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=50,
        callbacks=[lr_monitor, model_checkpoint, early_stopping],#, model_summary],
        logger=tensorboard_logger,
        gradient_clip_val=.1,
    )

    trainer.fit(model, datamodule)