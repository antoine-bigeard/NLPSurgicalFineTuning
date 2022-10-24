from distutils.command.build_ext import extension_name_re
from sys import prefix
from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from src.pytorch_utils import trans_window
import time

# from utils import transform_window

prefix_path_s3 = "s3://q3-techno-dev/"


class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx: int) -> Tuple[np.array, int]:

        pass
