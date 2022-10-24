from cgi import test
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
from PIL import Image

from src.data.dataset.dataset import MyDataset, PredictDataset
from utils import sample_idxs, compute_stats, save_idxs
import pandas as pd
import os


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_df_train: str,
        path_df_val: str,
        path_df_test: str,
        path_df_predict: str,
        selected_classes: list = [],
        batch_size: int = 4,
        shuffle: bool = False,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        sampler = None,
    ):
        self.selected_classes = selected_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.path_df_train = path_df_train
        self.path_df_val = path_df_val
        self.path_df_test = path_df_test
        self.path_df_predict = path_df_predict
        self.sampler = sampler

    def prepare_data(self):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.predict_df = None

    def setup(self, stage="fit"):  # stage = fit or test or predict
        self.prepare_data()

        if stage == "fit":
            self.train_dataset = MyDataset(
                self.train_df,
            )
            self.val_dataset = MyDataset(
                self.val_df,
            )

        elif stage == "test":
            test_transform_img = self.transforms_img["val"]
            if self.save_dfs:
                self.test_df.to_csv(os.path.join(self.path_logs, "test_df.csv"))
            self.test_dataset = MyDataset(
                self.test_df,
                test_transform_img,
            )

        elif stage == "predict":
            if self.save_dfs:
                self.predict_df.to_csv(os.path.join(self.path_logs, "predict_df.csv"))
            self.predict_dataset = PredictDataset(
                self.predict_df, selected_categories=self.selected_categories
            )

        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            sampler=ImbalancedDatasetSampler(self.train_dataset)
            if self.sampler == "imbalanced_dataset_sampler"
            else None,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            sampler=None,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            sampler=None,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            sampler=None,
            num_workers=self.num_workers,
        )
