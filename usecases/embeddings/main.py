import sys
import time

import ee
import geemap
import torch

sys.path.append("/Users/felix/Projects/geospatial_foundation_models/clay/model")
from dataset import Sentinel2DataModule

from model import Embedder

# from src.module import ClayMAEModule


ee.Initialize()


def load_point_dataset():
    asset_id_random_points_land = (
        "projects/gem-eth-analysis/assets/felix/misc/random_points_land_10000"
    )
    fc = ee.FeatureCollection(asset_id_random_points_land)
    points_df = geemap.ee_to_gdf(fc)
    return points_df


def main():
    # Load the point dataset
    points = load_point_dataset()

    ckpt_path = "checkpoints/v1.5/clay_v1.5.ckpt"

    # Load the model
    model = Embedder(ckpt_path=ckpt_path)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    model.eval()

    # check if cuda is available
    data_module = Sentinel2DataModule(
        raster_path="/Users/felix/Projects/geospatial_foundation_models/clay/model/data/sdms/merged_cog/s2_sr_global_SUM-2019-2024_epsg-4326_1000.tif",
        points_gdf=points,
        batch_size=64,
        window_size=128,
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    )

    data_module.setup()

    embeddings = []

    # Initialize timers
    total_data_loading_time = 0.0
    total_inference_time = 0.0
    i = 1
    start_data_loading = time.time()
    # iterate over dataset to check
    for batch in data_module.train_dataloader(num_workers=1):
        print(f"Batch {i} / {len(data_module.train_dataloader())}")
        # Measure data loading time

        batch = {k: v.to(device) for k, v in batch.items()}
        end_data_loading = time.time()

        data_loading_time = end_data_loading - start_data_loading
        total_data_loading_time += data_loading_time
        print(f"Data Loading Time for Batch {i}: {data_loading_time:.4f} seconds")

        # Measure inference time
        start_inference = time.time()
        output = model(batch)
        end_inference = time.time()

        inference_time = end_inference - start_inference
        total_inference_time += inference_time
        print(f"Inference Time for Batch {i}: {inference_time:.4f} seconds")

        embeddings.append(output)
        i += 1
        start_data_loading = time.time()

    # stack together the embeddings
    embeddings = torch.cat(embeddings)

    # convert to numpy
    embeddings = embeddings.detach().numpy()
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    print(f"Embeddings shape: {embeddings_np.shape}")


if __name__ == "__main__":
    main()
