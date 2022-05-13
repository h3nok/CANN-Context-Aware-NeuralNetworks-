import os
import glob
from PIL import Image
import numpy as np
import tqdm


def _encode_label(label):
    if label.lower() == 'triangle':
        return [0]
    elif label.lower() == 'circle':
        return [1]
    elif label.lower() == 'square':
        return [2]


class ShapesDataset:
    def __init__(self,
                 dataset_path: str = ''):
        assert os.path.exists(dataset_path)
        self.dataset_path = dataset_path

        self._basic_shapes_data = []
        self._basic_shapes_label = []

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.file_pbar = tqdm.tqdm(glob.glob(self.dataset_path + "/*.png"),
                                   desc='Preparing dataset')
        self._prepare()

    def _prepare(self, basic=True):
        for file in self.file_pbar:
            if basic:
                if 'Triangle' in file:
                    img = np.array(Image.open(file).resize((32, 32))).astype(dtype=np.uint8)
                    lbl = np.array(_encode_label('Triangle')).astype(dtype=np.uint8)
                    self._basic_shapes_data.append(img)
                    self._basic_shapes_label.append(lbl)
                elif 'Circle' in file:
                    img = np.array(Image.open(file).resize((32, 32))).astype(dtype=np.uint8)
                    lbl = np.array(_encode_label('Circle')).astype(dtype=np.uint8)
                    self._basic_shapes_data.append(img)
                    self._basic_shapes_label.append(lbl)
                elif 'Square' in file:
                    img = np.array(Image.open(file).resize((32, 32))).astype(dtype=np.uint8)
                    lbl = np.array(_encode_label('Square')).astype(dtype=np.uint8)
                    self._basic_shapes_data.append(img)
                    self._basic_shapes_label.append(lbl)
                else:
                    continue

        assert len(self._basic_shapes_label) == len(self._basic_shapes_label)
        import random

        random.shuffle(self._basic_shapes_label)
        random.shuffle(self._basic_shapes_data)
        split = int(len(self._basic_shapes_label) * 0.8)
        self.x_train = self._basic_shapes_data[:split]
        self.y_train = self._basic_shapes_label[:split]
        self.x_test = self._basic_shapes_data[split:]
        self.y_test = self._basic_shapes_label[split:]

        print(f"Train: {len(self.x_train)}")
        print(f"Test: {len(self.x_test)}")

        print(self.x_train[0].shape)

    @property
    def train_dataset(self):
        return self.x_train, self.y_train

    @property
    def test_dataset(self):
        return self.x_test, self.y_test


if __name__ == '__main__':
    path = r'C:\deepclo\dataset\shapes'

    shapes_ds = ShapesDataset(dataset_path=path)
