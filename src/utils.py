import numpy as np
from skimage.measure import label, regionprops
import cv2
from PIL import Image
from const import COLOR_MAP


def split_image_label_column(df):
    """
    Split column to get label and image name.
    :param df: DataFrame
    :return: DataFrame
    """
    df["label"] = df["Image_Label"].apply(lambda x: x.split("_")[1])
    df["image_id"] = df["Image_Label"].apply(lambda x: x.split("_")[0])

    df = df.drop(columns=["Image_Label"])
    return df


def get_mask_from_rle(mask_rle, width, height):
    """
    Get mask from the run-length encoding on the pixel values.
    :param mask_rle: string
    :param width: int
    :param height: int
    :return: np.array
    """
    rle_numbers = [int(num_string) for num_string in mask_rle.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    img = np.zeros(width * height, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index + length] = 100
    img = img.reshape(width, height)
    np_mask = img.T
    np_mask = np.clip(np_mask, 0, 1)
    return np_mask


def get_bounding_boxes_from_mask(mask):
    """
    # Get bounding box from mask.
    :param mask: np.array
    :return: tuple
    """
    label_mask = label(mask)
    props = regionprops(label_mask)
    bbox = [region.bbox for region in props if region.area > 50]
    return bbox


def get_image_with_bboxes(image, label):
    """
    Get image with bounding boxes
    :param image: PIL image
    :param label: tuple
    :return: PIL image
    """
    opencv_image = np.array(image)

    label_name = label[0]
    bboxes = label[1]
    for bbox in bboxes:
        start_point = (bbox[1], bbox[0])
        end_point = (bbox[3], bbox[2])

        text_point = (bbox[1] + 50, bbox[0] + 50)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = COLOR_MAP.get(label_name)
        opencv_image = cv2.rectangle(opencv_image, start_point, end_point,
                                     color, 8)

        opencv_image = cv2.putText(opencv_image, label_name, text_point,
                                   font, 1.5, color, 3, lineType=cv2.LINE_AA)

    image = Image.fromarray(opencv_image, 'RGB')

    return image
    # plt.imshow(image)