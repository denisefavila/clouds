import os

COLOR_MAP = {"Flower": (255, 0, 0),
             "Sugar": (0, 255, 0),
             "Gravel": (0, 0, 255),
             "Fish": (255, 255, 0)
            }

CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(ROOT_DIR, "understanding_cloud_organization/train_images")
TEST_PATH = os.path.join(ROOT_DIR, "understanding_cloud_organization/test_images")

TRAIN_CSV = os.path.join(ROOT_DIR, "understanding_cloud_organization/train.csv")
