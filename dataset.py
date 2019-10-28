import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from const import TRAIN_PATH, TEST_PATH
from utils import get_mask_from_rle, get_bounding_boxes_from_mask


class Cloud(Dataset):
    def __init__(self, df, datatype='train', img_ids=None, transform=None):
        self.df = df
        if datatype == 'train':
            self.data_folder = TRAIN_PATH
        else:
            self.data_folder = TEST_PATH

        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.df['image_id'][idx]
        image_label = self.df['label'][idx]
        mask_rle = self.df['EncodedPixels'][idx]

        image_path = os.path.join(self.data_folder, image_name)
        image = Image.open(image_path)
        width, height = image.size

        try:
            mask = get_mask_from_rle(mask_rle, width, height)
        except AttributeError:
            mask = np.zeros((height, width))

        bbox = get_bounding_boxes_from_mask(mask)

        sample = {'image': image,
                  'label': [image_label, bbox]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.df.shape[0]
