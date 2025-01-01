import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import FrameFolderDataset, unzip, MultiZipDataset

class FrameFolderDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=64,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def setup(self, stage=None):

        # Create the output folder if it doesn't exist
        output_dir = os.path.join(self.dataset_path, 'tmp/')

        # check if the output dir exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            unzip(self.dataset_path, output_dir=output_dir)

        self.train_dataset = FrameFolderDataset(dataset_path=os.path.join(output_dir,'train'), stage='train')
        self.val_dataset = FrameFolderDataset(dataset_path=os.path.join(output_dir,'val'), stage='val')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
class ZipDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_df,
                 num_classes=None,
                 batch_size=64,
                 num_workers=1,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_df = dataset_df
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def setup(self, stage=None):

        train_zip_files = self.dataset_df[self.dataset_df['split'] == 'train']['paths'].values
        print("train_zip_files", train_zip_files)
        val_zip_files = self.dataset_df[self.dataset_df['split'] == 'val']['paths'].values

        self.train_dataset = MultiZipDataset(train_zip_files, stage='train', num_classes=self.num_classes)
        self.val_dataset = MultiZipDataset(val_zip_files, stage='val', num_classes=self.num_classes)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

if __name__ == '__main__':
    # Usage example:
    import pandas as pd
    if 0:
        zip_df = pd.read_csv('ssl_zipfiles.csv')
        datamodule = ZipDataModule(dataset_df=zip_df, num_classes=None)
    else:
        zip_df = pd.read_csv('supervised_zipfiles.csv')
        datamodule = ZipDataModule(dataset_df=zip_df, num_classes=2)
