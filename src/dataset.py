import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from const import TRAIN_PATH, TEST_PATH, CLASSES
from src.utils import get_mask_from_rle, get_bounding_boxes_from_mask


class Cloud(Dataset):
    def __init__(self, df, datatype='train', transform=None):

        self.df = df
        if datatype == 'train':
            self.data_folder = TRAIN_PATH
        else:
            self.data_folder = TEST_PATH

        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.df['image_id'][idx]
        image_labels = list(self.df[CLASSES].iloc[idx])
        #masks_rle = list(self.df['encoded'][idx].values())
        masks_rle = [self.df['encoded'][idx].get(klass, np.nan)
                    for klass in CLASSES]  #I'm still think about both solutions

        image_path = os.path.join(self.data_folder, image_name)
        image = Image.open(image_path)
        width, height = image.size

        masks = []
        for mask_rle in masks_rle:
            try:
                mask = get_mask_from_rle(mask_rle, width, height)
            except AttributeError:
                mask = np.zeros((height, width))
            masks.append(mask)

        bboxes = [get_bounding_boxes_from_mask(mask) for mask in masks]

        sample = {'image': image,
                  'boxes': bboxes,
                  'labels': image_labels
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.df.shape[0]
