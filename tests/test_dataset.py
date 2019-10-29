import pytest
import pandas as pd
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile

from src.dataset import Cloud


@pytest.fixture()
def data_params():
    """Return a map from params."""
    return {
        'train_csv_with_cloud_formation': {"label": ["Fish", "Flower"],
                                           "image_id": ["0a1b596.jpg", "0a5d418.jpg"],
                                           "EncodedPixels": ["264918 937 266318 937 267718 937",
                                                             "264928 931 262318 947 267118 917"]},
        'train_csv_without_cloud_formation': {"label": ["Fish", "Flower"],
                                              "image_id": ["0a1b596.jpg", "0a5d418.jpg"],
                                              "EncodedPixels": [np.nan, np.nan]},
        'test_csv_with_cloud_formation': {"label": ["Fish", "Flower"],
                                          "image_id": ["0a0f81b.jpg", "0a2c1c4.jpg"],
                                          "EncodedPixels": ["264918 937 266318 937 267718 937",
                                                            "264928 931 262318 947 267118 917"]},
    }


@pytest.fixture(params=[
    'train_csv_with_cloud_formation',
    'train_csv_without_cloud_formation',
    'test_csv_with_cloud_formation'
])
def data(data_params, request):
    """Create a Sushi instance based on recipes."""
    name = request.param
    return pd.DataFrame.from_dict(data_params[name])


@pytest.mark.parametrize('data, bbox_qty, data_type', [
    ("train_csv_with_cloud_formation", 1, "train"),
    ("train_csv_without_cloud_formation", 0, "train"),
    ("test_csv_with_cloud_formation", 1, "test"),
], indirect=["data"])
def test_get_example(data, bbox_qty, data_type):
    train_dataset = Cloud(df=data,
                          datatype=data_type)
    get_one_example = train_dataset[0]

    assert isinstance(get_one_example['image'], JpegImageFile)
    assert get_one_example['label'][0] == data.iloc[0]["label"]

    assert len(get_one_example['label'][1]) == bbox_qty


def test_get_len(data):
