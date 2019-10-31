import pytest
import pandas as pd
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile

from const import CLASSES
from src.dataset import Cloud
from src.utils import split_image_label_column


@pytest.fixture()
def data_params():
    """Return a map from params."""
    return {
        'train_csv_with_cloud_formation': {"Image_Label": ["0a1b596.jpg_Fish", "0a1b596.jpg_Flower"],
                                           "EncodedPixels": ["264918 937 266318 937 267718 937",
                                                             "264918 937 266318 937 267718 937"]},
        
        'train_csv_without_cloud_formation': {"Image_Label": ["0a1b596.jpg_Fish", "0a1b596.jpg_Flower"],
                                              "EncodedPixels": [np.nan, np.nan]},

        'test_csv_with_cloud_formation': {"Image_Label": ["0a0f81b.jpg_Fish", "0a2c1c4.jpg_Flower"],
                                          "EncodedPixels": ["264918 937 266318 937 267718 937",
                                                            "264928 931 262318 947 267118 917"]},
    }


@pytest.fixture(params=[
    'train_csv_with_cloud_formation',
    'train_csv_without_cloud_formation',
    'test_csv_with_cloud_formation'
])
def data(data_params, request):
    """Create dataframe."""
    name = request.param
    return pd.DataFrame.from_dict(data_params[name])


@pytest.mark.parametrize('data, bbox_qty, data_type', [
    ("train_csv_with_cloud_formation", 1, "train"),
    ("train_csv_without_cloud_formation", 0, "train"),
    ("test_csv_with_cloud_formation", 1, "test"),
], indirect=["data"])
def test_get_example(data, bbox_qty, data_type):

    data = split_image_label_column(data)
    train_dataset = Cloud(df=data,
                          datatype=data_type)

    example = 0
    get_one_example = train_dataset[example]

    assert isinstance(get_one_example['image'], JpegImageFile)
    assert len(get_one_example['boxes']) == 4
    assert len(get_one_example['labels']) == 4
    assert np.alltrue(get_one_example['labels'] == data.iloc[example][CLASSES].values)


