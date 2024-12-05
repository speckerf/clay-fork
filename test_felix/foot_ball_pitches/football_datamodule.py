import geopandas as gpd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio
import torch
import tqdm
import yaml
from box import Box
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

BANDS_TO_INDICES = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B11": 10,
    "B12": 11,
}


class Sentinel2Dataset(Dataset):
    def __init__(
        self,
        raster_path,
        points_gdf,
        labels,
        window_size=64,
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        transform=None,
    ):
        """
        Args:
            raster_path (str): Path to the Sentinel-2 COG.
            points_gdf (GeoDataFrame): GeoDataFrame containing point locations.
            labels (list): List of corresponding labels for each point.
            window_size (int): Size of the window around each point (default: 64).
            transform (callable, optional): Transformations to apply to the data.
        """
        self.raster_path = raster_path
        self.points_gdf = points_gdf
        self.labels = labels
        self.window_size = window_size
        self.transform = transform
        self.bands = bands

        # Open the raster file once
        # self.raster = rasterio.open(raster_path)
        self.raster = None

    def __len__(self):
        return len(self.points_gdf)

    def __getitem__(self, idx):

        # Open the raster if not already opened
        if self.raster is None:
            self.raster = rasterio.open(self.raster_path)
        # Get point and label
        point = self.points_gdf.iloc[idx]
        label = self.labels[idx]

        # Convert point geometry to raster pixel coordinates
        row, col = self.raster.index(point.geometry.x, point.geometry.y)

        # Calculate window bounds
        half_window = self.window_size // 2
        window = Window(
            col_off=col - half_window,
            row_off=row - half_window,
            width=self.window_size,
            height=self.window_size,
        )

        # Read the window from the raster
        window_data = self.raster.read(
            window=window,
            indexes=[BANDS_TO_INDICES[band] for band in self.bands],
        )  # Shape: (bands, window_size, window_size)

        # Handle edge cases (e.g., points near the raster boundaries)
        if (
            window_data.shape[1] != self.window_size
            or window_data.shape[2] != self.window_size
        ):
            padded_window = np.zeros(
                (window_data.shape[0], self.window_size, self.window_size),
                dtype=window_data.dtype,
            )
            padded_window[
                :,
                : window_data.shape[1],
                : window_data.shape[2],
            ] = window_data
            window_data = padded_window

        # Apply transformations if provided
        if self.transform:
            window_data = self.transform(window_data)

        sample = {
            "pixels": torch.tensor(window_data, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for lat/lon information
        }

        return sample


class FootballPitchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        raster_path,
        points_gdf,
        labels,
        window_size=64,
        batch_size=16,
        num_workers=4,
        val_split=0.2,
        metadata_path="configs/metadata.yaml",
    ):
        """
        Initialize the data module.

        Args:
            raster_path (str): Path to the Sentinel-2 COG.
            points_gdf (GeoDataFrame): GeoDataFrame containing point locations.
            labels (list): List of corresponding labels for each point.
            window_size (int): Size of the window around each point.
            batch_size (int): Batch size for the dataloaders.
            num_workers (int): Number of workers for data loading.
            val_split (float): Fraction of data to use for validation.
            transform (callable, optional): Transformations to apply to the data.
        """
        super().__init__()
        self.raster_path = raster_path
        self.points_gdf = points_gdf
        self.labels = labels
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        metadata = Box(yaml.safe_load(open(metadata_path)))["sentinel-2-l2a"]
        mean = list(metadata.bands.mean.values())
        std = list(metadata.bands.std.values())

        self.trn_tfm = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.Normalize(mean, std),
            ]
        )
        self.val_tfm = v2.Compose([v2.Normalize(mean, std)])

        self.train_points = None
        self.val_points = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """
        Split the GeoDataFrame into training and validation sets and create datasets.
        """
        # Split the data into train and val sets
        train_gdf, val_gdf, train_labels, val_labels = train_test_split(
            self.points_gdf, self.labels, test_size=self.val_split, random_state=42
        )

        # Track train and val points
        self.train_points = train_gdf
        self.val_points = val_gdf

        # Create datasets
        self.train_dataset = Sentinel2Dataset(
            raster_path=self.raster_path,
            points_gdf=train_gdf,
            labels=train_labels,
            window_size=self.window_size,
            transform=self.trn_tfm,
        )
        self.val_dataset = Sentinel2Dataset(
            raster_path=self.raster_path,
            points_gdf=val_gdf,
            labels=val_labels,
            window_size=self.window_size,
            transform=self.val_tfm,
        )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def get_train_points(self):
        """
        Returns the GeoDataFrame of training points.
        """
        return self.train_points

    def get_val_points(self):
        """
        Returns the GeoDataFrame of validation points.
        """
        return self.val_points
