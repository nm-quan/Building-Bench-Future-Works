"""Vendored, line-for-line ports of the official BuildingsBench preprocessing
transforms, so this study's preprocessing is provably identical to the paper's
without depending on the `buildings-bench` PyPI package (confirmed broken as of
this writing: its sdist is missing requirements.txt and its `rliable` dependency
fails to build a wheel under current setuptools -- see the repo's own install
notes about needing a hand-downloaded faiss-gpu wheel, which is the same class
of packaging fragility).

Source: https://github.com/NatLabRockies/BuildingsBench (BSD-3-Clause,
Copyright (c) 2023, National Renewable Energy Laboratory), file
`buildings_bench/transforms.py`, classes BoxCoxTransform, StandardScalerTransform,
TimestampTransform. Reproduced here verbatim (torch/numpy dependency signature
unchanged) rather than imported, so this module has no dependency on the
upstream package's broken packaging.
"""
import pickle as pkl
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import torch


class BoxCoxTransform:
    """Computes and applies the Box-Cox transform to load data. Matches the
    official `BoxCoxTransform`: sklearn PowerTransformer(method='box-cox',
    standardize=True), fit globally (not per-building) on the pooled load
    values, epsilon-offset by 1e-6 before fitting/transforming (Box-Cox
    requires strictly positive input)."""

    def __init__(self, max_datapoints: int = 1_000_000):
        self.boxcox = None
        self.max_datapoints = max_datapoints

    def train(self, data: np.ndarray) -> None:
        self.boxcox = preprocessing.PowerTransformer(method="box-cox", standardize=True)
        data = data.flatten().reshape(-1, 1)
        if data.shape[0] > self.max_datapoints:
            data = data[np.random.choice(data.shape[0], self.max_datapoints, replace=False)]
        self.boxcox.fit_transform(1e-6 + data)

    def save(self, output_path: Path) -> None:
        with open(Path(output_path) / "boxcox.pkl", "wb") as f:
            pkl.dump(self.boxcox, f)

    def load(self, saved_path: Path) -> None:
        p = Path(saved_path)
        f = p / "boxcox.pkl" if p.is_dir() else p
        with open(f, "rb") as fh:
            self.boxcox = pkl.load(fh)

    def transform(self, sample: np.ndarray) -> np.ndarray:
        init_shape = sample.shape
        return self.boxcox.transform(1e-6 + sample.flatten().reshape(-1, 1)).reshape(init_shape)

    def undo_transform(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(sample, torch.Tensor)
        if is_tensor:
            device = sample.device
            sample = sample.cpu().numpy()
        init_shape = sample.shape
        sample = self.boxcox.inverse_transform(sample.flatten().reshape(-1, 1)).reshape(init_shape)
        if is_tensor:
            sample = torch.from_numpy(sample).to(device)
        return sample


class StandardScalerTransform:
    """Standardizes data by removing the mean and scaling to unit variance.
    Used for the 7 weather channels, one instance per channel, matching the
    official `StandardScalerTransform`."""

    def __init__(self, max_datapoints: int = 1_000_000, device: str = "cpu"):
        self.mean_ = None
        self.std_ = None
        self.max_datapoints = max_datapoints
        self.device = device

    def train(self, data: np.ndarray) -> None:
        data = data.flatten().reshape(-1, 1)
        if data.shape[0] > self.max_datapoints:
            data = data[np.random.choice(data.shape[0], self.max_datapoints, replace=False)]
        self.mean_ = torch.from_numpy(np.array([np.mean(data)])).float().to(self.device)
        self.std_ = torch.from_numpy(np.array([np.std(data)])).float().to(self.device)

    def save(self, output_path: Path) -> None:
        mean_ = self.mean_.cpu().numpy().reshape(-1)
        std_ = self.std_.cpu().numpy().reshape(-1)
        np.save(Path(output_path) / "standard_scaler.npy", np.array([mean_, std_]))

    def load(self, saved_path: Path) -> None:
        p = Path(saved_path)
        f = p / "standard_scaler.npy" if p.is_dir() else p
        x = np.load(f)
        self.mean_ = torch.from_numpy(np.array([x[0]])).float().to(self.device)
        self.std_ = torch.from_numpy(np.array([x[1]])).float().to(self.device)

    def transform(self, sample: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float().to(self.device)
        return (sample - self.mean_) / self.std_

    def undo_transform(self, sample: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float().to(self.device)
        return self.std_ * sample + self.mean_


class TimestampTransform:
    """Extracts [day_of_year, day_of_week, hour_of_day] from a timestamp
    series/index, each linearly scaled to [-1, 1]. Matches the official
    `TimestampTransform` exactly -- NOT sin/cos encoded, which is the
    deviation the earlier draft of this study had."""

    def __init__(self, is_leap_year: bool = False) -> None:
        self.day_year_normalization = 365 if is_leap_year else 364
        self.hour_of_day_normalization = 23
        self.day_of_week_normalization = 6

    def transform(self, timestamp_series) -> np.ndarray:
        if isinstance(timestamp_series, pd.DatetimeIndex):
            timestamp_series = timestamp_series.to_series()
        timestamp_series = pd.to_datetime(timestamp_series)
        day_of_week = timestamp_series.dt.dayofweek
        day_of_year = timestamp_series.dt.dayofyear
        hour_of_day = timestamp_series.dt.hour
        time_features = np.stack(
            [
                day_of_year / self.day_year_normalization,
                day_of_week / self.day_of_week_normalization,
                hour_of_day / self.hour_of_day_normalization,
            ],
            axis=1,
        ).astype(np.float32)
        return time_features * 2 - 1
