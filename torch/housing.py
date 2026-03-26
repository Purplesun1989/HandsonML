 # ./torch/housing.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from typing import Tuple

class HousingValueModel(nn.Module):
    def __init__(self, input: int, output: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, output)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    

def read_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    raw_df = pd.read_csv(csv_path)
    df = raw_df.dropna(how="all")
    return df

def get_tensor(raw_df: pd.DataFrame, y_column_name: str, ocean_column_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.get_dummies(raw_df, columns = [ocean_column_name])
    X = df.drop(columns=y_column_name)
    y = df[y_column_name].values.reshape(-1, 1) # type: ignore

    x_scaled = StandardScaler()
    y_scaled = StandardScaler()
    X = x_scaled.fit_transform(X)
    y = y_scaled.fit_transform(y) # type: ignore

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

def create_dataloader(X_tensor, y_tensor) -> Tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=64)

    return train_loader, val_loader
    

if __name__ == "__main__":
    csv_path = r"Chapter_2/datasets/housing/housing.csv"
    y_column_name = r"median_house_value"
    ocean_column_name = r"ocean_proximity"
    raw_df = read_data(csv_path)
    X_tensor, y_tensor = get_tensor(raw_df, y_column_name, ocean_column_name)
    print(X_tensor[:5])



    