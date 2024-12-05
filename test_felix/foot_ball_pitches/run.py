import sys

sys.path.append("/Users/felix/Projects/geospatial_foundation_models/clay/model")

import ee
import geemap
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from test_felix.foot_ball_pitches.football_datamodule import FootballPitchDataModule
from test_felix.foot_ball_pitches.football_model import (
    ConfusionMatrixCallback,
    FootballPitchModel,
)

ee.Initialize()


def get_football_pitches():
    asset_id = "projects/ee-speckerfelix/assets/clay/pitch_classification"
    fc = ee.FeatureCollection(asset_id)
    gdf = geemap.ee_to_gdf(fc)
    return gdf


def main():
    points_gdf = get_football_pitches()
    raster_path = "data/football_pitches/zurich_cog.tif"
    labels = points_gdf["label"].apply(lambda x: 1 if x == "pitch" else 0).tolist()

    # Create the dataset
    data_module = FootballPitchDataModule(
        raster_path=raster_path,
        points_gdf=points_gdf,
        labels=labels,
        window_size=64,
        batch_size=64,
        num_workers=4,
        val_split=0.2,
    )

    # Call setup to initialize datasets
    data_module.setup()

    print(f"Train Dataset: {data_module.train_dataset}")
    print(f"Val Dataset: {data_module.val_dataset}")

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # for batch_idx, data in enumerate(train_dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"Pixels shape: {data['pixels'].shape}")
    #     print(f"Lable shape: {data['label'].shape}")
    #     print(f"Time shape: {data['time'].shape}")
    #     print(f"LatLon shape: {data['latlon'].shape}")

    #     data
    #     break  # Test with only the first batch

    ckpt_path = "checkpoints/v1.5/clay_v1.5.ckpt"
    model = FootballPitchModel(num_classes=1, ckpt_path=ckpt_path)

    # # check if encoder is frozen
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    logger = WandbLogger(project="football-pitches", entity="speckerf")

    confmat_callback = ConfusionMatrixCallback(task="binary")
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        strategy="auto",
        logger=logger,
        callbacks=[confmat_callback],
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
