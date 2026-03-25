# ./torch/housing.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

class HousingValueModel(nn.Module):
    def __init__(self, input: int, output: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
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

if __name__ == "__main__":
    csv_path = r"housing.csv"
    raw_df = read_data(csv_path)
    