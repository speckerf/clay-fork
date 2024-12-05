import math

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
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
    "B11": 9,
    "B12": 10,
}  # check using gdalinfo on the raster file


# Prep datetimes embedding using a normalization function from the model code.
def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


# Prep lat/lon embedding using the
def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return ((math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon)))


class Sentinel2Dataset(Dataset):
    def __init__(
        self,
        raster_path,
        points_gdf,
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

        location_encoded = normalize_latlon(lat=point.geometry.y, lon=point.geometry.x)

        sample = {
            "pixels": torch.from_numpy(window_data.astype(np.float32)),
            "time": torch.zeros(4, dtype=torch.float32),
            "latlon": torch.tensor(
                np.hstack((location_encoded[0], location_encoded[1])),
                dtype=torch.float32,
            ),
        }

        return sample


class Sentinel2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        raster_path,
        points_gdf,
        batch_size=32,
        window_size=64,
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        transform=None,
    ):
        super().__init__()
        self.raster_path = raster_path
        self.points_gdf = points_gdf
        self.batch_size = batch_size
        self.window_size = window_size
        self.bands = bands
        self.transform = transform

    def setup(self, stage=None):
        self.dataset = Sentinel2Dataset(
            self.raster_path,
            self.points_gdf,
            window_size=self.window_size,
            bands=self.bands,
            transform=self.transform,
        )

    def train_dataloader(self, num_workers=1):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        raise NotImplementedError
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        raise NotImplementedError
        return DataLoader(self.dataset, batch_size=self.batch_size)
