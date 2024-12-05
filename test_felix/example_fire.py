import math
import pickle
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
import yaml
from box import Box
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely import Point
from sklearn import decomposition, svm
from torchvision.transforms import v2

sys.path.append("/Users/felix/Projects/geospatial_foundation_models/clay/model")
from src.module import ClayMAEModule


def load_data():
    # Point over Monchique Portugal
    lat, lon = 37.30939, -8.57207

    # Dates of a large forest fire
    start = "2018-07-01"
    end = "2018-09-01"

    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    # Search the catalogue
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    print(f"Found {len(items)} items")

    # Extract coordinate system from first item
    epsg = items[0].properties["proj:epsg"]

    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    size = 256
    gsd = 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    # Retrieve the pixel values, for the bounding box in
    # the target projection. In this example we use only
    # the RGB and NIR bands.
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float32",
        rescale=False,
        fill_value=0,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )

    print(stack)

    stack = stack.compute()

    return stack


def main():
    # s2_stack = load_data()

    # # save s2_stack as pickle in tempfile
    # pickle.dump(s2_stack, open("test_felix/misc/s2_stack.pkl", "wb"))

    # load s2_stack from pickle
    s2_stack = pickle.load(open("test_felix/misc/s2_stack.pkl", "rb"))

    print(s2_stack)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # ckpt = "https://clay-model-ckpt.s3.amazonaws.com/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"
    # ckpt = "https://huggingface.co/made-with-clay/Clay/resolve/main/Clay_v0.1_epoch-24_val-loss-0.46.ckpt?download=true"
    # ckpt = "checkpoints/Clay_v0.1_epoch-24_val-loss-0.46.ckpt"
    ckpt = "checkpoints/clay_v1.5.ckpt"
    torch.set_default_device(device)

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()

    model = model.to(device)

    #### Prepare band metadata for passing it to the model
    # Extract mean, std, and wavelengths from metadata
    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
    mean = []
    std = []
    waves = []
    # Use the band names to get the correct values in the correct order.
    for band in s2_stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    # Prepare the normalization transform function using the mean and std values.
    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # Prep datetimes embedding using a normalization function from the model code.
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24

        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    datetimes = s2_stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    # Prep lat/lon embedding using the
    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180

        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    lat, lon = 37.30939, -8.57207
    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(s2_stack.data.astype(np.float32))
    pixels = transform(pixels)

    # Prepare additional information
    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(s2_stack.gsd.values, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    ## run the model
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

    # The first embedding is the class token, which is the
    # overall single embedding. We extract that for PCA below.
    embeddings = unmsk_patch[:, 0, :].cpu().numpy()

    # Run PCA
    pca = decomposition.PCA(n_components=1)
    pca_result = pca.fit_transform(embeddings)

    ## train a classifier head on the embeddings and use it to detect fires

    # Label the images we downloaded
    # 0 = Cloud
    # 1 = Forest
    # 2 = Fire
    labels = np.array([0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # Split into fit and test manually, ensuring we have all 3 classes in both sets
    fit = [0, 1, 3, 4, 7, 8, 9]
    test = [2, 5, 6, 10, 11]

    # Train a Support Vector Machine model
    clf = svm.SVC()
    clf.fit(embeddings[fit] + 100, labels[fit])

    # Predict classes on test set
    prediction = clf.predict(embeddings[test] + 100)

    # Perfect match for SVM
    match = np.sum(labels[test] == prediction)
    print(f"Matched {match} out of {len(test)} correctly")

    embeddings


if __name__ == "__main__":
    main()
