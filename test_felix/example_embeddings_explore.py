import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from lonboard import Map, PolygonLayer
from lonboard.colormap import apply_categorical_cmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def main():
    files = [
        "embeddings_ca_m_3712213_ne_10_060_20220518.gpq",
        "embeddings_ca_m_3712213_nw_10_060_20220518.gpq",
        "embeddings_ca_m_3712213_se_10_060_20220518.gpq",
        "embeddings_ca_m_3712213_sw_10_060_20220518.gpq",
        "embeddings_ca_m_3712214_sw_10_060_20220518.gpq",
        "embeddings_ca_m_3712221_ne_10_060_20220518.gpq",
        "embeddings_ca_m_3712221_nw_10_060_20220518.gpq",
        "embeddings_ca_m_3712221_sw_10_060_20220518.gpq",
        "embeddings_ca_m_3712222_sw_10_060_20220518.gpq",
        "embeddings_ca_m_3712229_ne_10_060_20220518.gpq",
        "embeddings_ca_m_3712230_nw_10_060_20220518.gpq",
        "embeddings_ca_m_3712212_ne_10_060_20220519.gpq",
        "embeddings_ca_m_3712212_nw_10_060_20220519.gpq",
        "embeddings_ca_m_3712212_se_10_060_20220519.gpq",
        "embeddings_ca_m_3712228_ne_10_060_20220519.gpq",
        "embeddings_ca_m_3712221_se_10_060_20220518.gpq",
        "embeddings_ca_m_3712222_nw_10_060_20220518.gpq",
        "embeddings_ca_m_3712220_ne_10_060_20220519.gpq",
        "embeddings_ca_m_3712229_nw_10_060_20220518.gpq",
        "embeddings_ca_m_3712214_nw_10_060_20220518.gpq",
        "marinas.geojson",
        "baseball.geojson",
    ]

    url_template = "https://huggingface.co/datasets/made-with-clay/classify-embeddings-sf-baseball-marinas/resolve/main/{filename}"

    for filename in files:
        dst = f"data/classify-embeddings-sf-baseball-marinas/{filename}"
        print(dst)
        if Path(dst).exists():
            continue
        with requests.get(url_template.format(filename=filename)) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)

    # Open embeddings DB
    embeddings = []
    for src in Path("data/classify-embeddings-sf-baseball-marinas/").glob("*.gpq"):
        gdf = gpd.read_parquet(src)
        embeddings.append(gdf)
    embeddings = pd.concat(embeddings)

    embeddings

    # Open marinas training data
    points = gpd.read_file(
        "data/classify-embeddings-sf-baseball-marinas/marinas.geojson"
    )

    # Uncomment this to use the baseball training dataset.
    # points = gpd.read_file(
    #     "../../data/classify-embeddings-sf-baseball-marinas/baseball.geojson"
    # )

    # Spatial join of training data with embeddings
    merged = embeddings.sjoin(points)
    print(f"Found {len(merged)} embeddings to train on")
    print(f"{sum(merged['class'])} marked locations")
    print(f"{len(merged) - sum(merged['class'])} negative examples")

    merged

    # Train a classifier
    # Extract X and y and split into test/train set
    X = np.array([dat for dat in merged["embeddings"].values])
    y = merged["class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit Random Forest classifier
    model = RandomForestClassifier()
    model = model.fit(X_train, y_train)

    # Make test prediction and evaluate
    pred = model.predict(X_test)
    print(f"Accuracy is {accuracy_score(y_test, pred)}")
    print(f"Precision is {precision_score(y_test, pred)}")
    print(f"Recall is {recall_score(y_test, pred)}")
    print(f"F1 is {f1_score(y_test, pred)} (harmonic mean of precision and recall)")

    # Make inference on entire embedding dataset
    X = np.array([x for x in embeddings["embeddings"]])
    predicted = model.predict(X)
    print(f"Found {np.sum(predicted)} locations")

    # Add inference to geopandas df and export
    result = embeddings[predicted.astype("bool")]
    result = result[["item_id", "geometry"]]


if __name__ == "__main__":
    main()
