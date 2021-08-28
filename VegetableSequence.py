import copy
import os
from absl import flags
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm import tqdm
import tensorflow as tf

from shuffle import shuffle_arrays, random_choice
from temporal_random_seed import TemporalRandomSeed
import myFlags
FLAGS = flags.FLAGS

class VegetableDataset():
    labels = ["red apple", "green apple", "banana", "red cabbage", "cabbage",
               "lemon", "mikan", "orange", "onion", "red onion",
              "green pepper", "red pepper", "yellow pepper", "persimmon", "tomato"]

    object_categories = len(labels)
    hsi_dims = 81
    
    def __init__(self, data_path):
        self.DATASET_ROOT_PATH = data_path


    def path(self, category_id=-1, instance_id=-1, sample_id=-1):
        p = self.DATASET_ROOT_PATH

        if category_id == -1:
            return p
        p = os.path.join(p, str(category_id).zfill(2))

        if instance_id == -1:
            return p
        p = os.path.join(p, str(instance_id).zfill(2))

        if sample_id == -1:
            return p
        p = os.path.join(p, str(sample_id).zfill(2))
        return p

    def get_categories(self):
        return list(range(1, self.object_categories + 1))

class VegetableSequence(Sequence):
    def __init__(self, batch_size, augmentations, dataset,
                 random_state=None,
                 isSeg=False,
                 isTest=False,
                 size=(224, 224),
                 nb_pixels=32,
                 instance_ids=[1, 2, 3],  # 4 for val, 5 for test
                 sample_ids=[1, 2]):
        self.batch_size = batch_size
        self.augment = augmentations
        self.random_state = random_state
        self.isSeg = isSeg
        self.isTest = isTest
        self.instance_ids = copy.deepcopy(instance_ids)
        self.sample_ids = copy.deepcopy(sample_ids)
        self.size = copy.deepcopy(size)
        self.nb_pixels = nb_pixels
        self.dataset = dataset

        self.nb_samples = self.dataset.object_categories * \
            len(self.instance_ids) * len(self.sample_ids)

        self.x = np.empty((self.nb_samples,) +
                          self.get_x_sample_shape(), dtype=self.get_x_dtype())
        self.y = np.empty((self.nb_samples,), dtype=self.get_y_dtype())
        self.y_seg = np.empty((self.nb_samples,) +
                              self.size, dtype=self.get_y_dtype())
        index = 0
        cats = self.dataset.get_categories()
        for cat_i, cat in enumerate(tqdm(cats)):
            for instance in self.instance_ids:
                for sample in self.sample_ids:
                    self.x[index] = self.load_sample(cat, instance, sample)
                    self.y[index] = cat_i
                    self.y_seg[index] = self.load_sample_seg(
                        cat, instance, sample)

                    index += 1
        nb_classes = self.dataset.object_categories
        self.y = to_categorical(self.y, num_classes=nb_classes)
        self.y_seg = to_categorical(self.y_seg, num_classes=2)

        if self.random_state is None:
            shuffle_arrays([self.x, self.y, self.y_seg], set_seed=1)

    def get_x_sample_shape(self):
        return self.size + (self.dataset.hsi_dims, )

    def get_x_dtype(self):
        return "uint16"

    def get_y_dtype(self):
        return "uint8"

    def load_sample(self, cat, instance, sample):
        path = self.dataset.path(cat, instance, sample)
        cache_fn = "224_224.m.rf.npy"
        cache_file_path = os.path.join(path, cache_fn)

        img = np.load(cache_file_path)
        img = img.astype(self.get_x_dtype())
        return img

    def load_sample_seg(self, cat, instance, sample):
        path = self.dataset.path(cat, instance, sample)
        cache_fn = "224_224.seg.npy"
        cache_file_path = os.path.join(path, cache_fn)

        seg = np.load(cache_file_path).astype(self.get_y_dtype())
        return seg

    def __len__(self):
        return int(np.ceil(float(self.nb_samples) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx %= self.__len__()
        batch_x = self.x[idx *
                         self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx *
                         self.batch_size:(idx + 1) * self.batch_size].astype("float32")
        batch_y_seg = self.y_seg[idx *
                                 self.batch_size:(idx + 1) * self.batch_size].astype("float32")

        seed = None if self.random_state is None else idx * self.random_state
        with TemporalRandomSeed(seed):
            if self.isTest:
                xs = np.empty_like(batch_x, dtype="float32")
                ys = np.empty_like(batch_y, dtype="float32")
                for i, (x, y) in enumerate(zip(batch_x, batch_y)):
                    augumented = self.augment(image=x)
                    xs[i] = augumented["image"]
                    ys[i] = y
            else:
                if self.isSeg:
                    xs = np.empty_like(batch_x, dtype="float32")
                    ys = np.empty_like(batch_y_seg, dtype="float32")
                    for i, (x, y_seg) in enumerate(zip(batch_x, batch_y_seg)):
                        augumented = self.augment(image=x, mask=y_seg)
                        xs[i] = augumented["image"]
                        ys[i] = augumented["mask"]
                else:
                    xs = []
                    ys = []
                    for i, (x, y, y_seg) in enumerate(zip(batch_x, batch_y, batch_y_seg)):
                        augumented = self.augment(image=x, mask=y_seg)
                        aug_x = augumented["image"]
                        aug_y_seg = augumented["mask"]
                        object_pixels = aug_x[np.argmax(
                            aug_y_seg, axis=-1) > 0]
                        xs.append(random_choice(object_pixels,
                                                self.nb_pixels, set_seed=seed))
                        ys.append(y)
                    xs = np.array(xs)
                    ys = np.array(ys)
        return xs, ys

    def on_epoch_end(self):
        if self.random_state is None:
            shuffle_arrays([self.x, self.y, self.y_seg], set_seed=1)