from src.const import TRAIN_CSV
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import split_image_label_column, get_image_with_bboxes
from src.dataset import Cloud


def main():

    # Load DataFrame with train labels
    train_df = split_image_label_column(pd.read_csv(TRAIN_CSV))
    print('There are {} images in the train set.'.format(len(train_df)))

    train_dataset = Cloud(df=train_df, datatype='train')

    # Plot figure with bounding box
    plt.figure(figsize=(20, 10))
    image = get_image_with_bboxes(**train_dataset[0])

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
