import sys

sys.path.append("/Users/felix/Projects/geospatial_foundation_models/clay/model")

import random

import ee
import geemap
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from test_felix.sdm_test.datamodule import SDMsDataModule
from test_felix.sdm_test.model import SMDsModel

ee.Initialize()


def get_test_gbif_data(num_mock_species=5):
    asset_id_random_points_land = (
        "projects/gem-eth-analysis/assets/felix/misc/random_points_land_10000"
    )
    fc = ee.FeatureCollection(asset_id_random_points_land)
    points_df = geemap.ee_to_gdf(fc)

    # assign random integer from 0 to num_mock_species-1 to each point
    points_df["species"] = pd.Series(
        (random.randint(0, num_mock_species - 1) for _ in range(len(points_df)))
    )

    return points_df


def main():
    num_species = 5
    points_gdf = get_test_gbif_data(num_mock_species=num_species)
    raster_path = "data/sdms/merged_cog/s2_sr_global_SUM-2019-2024_epsg-4326_1000.tif"

    # create one hot encoding for the labels
    labels = np.array(points_gdf["species"])
    one_hot_labels = np.eye(num_species)[labels]

    # Create the dataset
    data_module = SDMsDataModule(
        raster_path=raster_path,
        points_gdf=points_gdf,
        labels=one_hot_labels,
        window_size=64,
        batch_size=64,
        num_workers=1,
        val_split=0.2,
    )

    # Call setup to initialize datasets
    data_module.setup()

    print(f"Train Dataset: {data_module.train_dataset}")
    print(f"Val Dataset: {data_module.val_dataset}")

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # test iterate over train dataloader
    for batch_idx, data in enumerate(train_dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Pixels shape: {data['pixels'].shape}")
        print(f"Lable shape: {data['label'].shape}")
        print(f"Time shape: {data['time'].shape}")
        print(f"LatLon shape: {data['latlon'].shape}")

    ckpt_path = "checkpoints/v1.5/clay_v1.5.ckpt"
    model = SMDsModel(num_classes=num_species, ckpt_path=ckpt_path)

    # check if encoder is frozen
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    logger = WandbLogger(project="sdms-test", entity="speckerf")

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        strategy="auto",
        logger=logger,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
