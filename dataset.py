import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import os, glob
import rawpy


class MyDataset(Dataset):
    def __init__(self, raw_dir, proc_dir, transform=None, transform2=None):
        self.transform = transform
        self.transform2 = transform2
        self.raw_dataset = glob.glob(raw_dir + "*.*")
        # quick solution for same named files to match
        self.raw_dataset.sort()
        self.proc_dataset = glob.glob(proc_dir + "*.*")
        self.proc_dataset.sort()
        assert len(self.raw_dataset) == len(self.proc_dataset), 'inputs and standards sizes must be equal'

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        #on my local machine "with" get a memory error
        raw = rawpy.imread(self.raw_dataset[idx])
        raw_ = raw.raw_image
        # raw.close()
        proc = io.imread(self.proc_dataset[idx])

        if self.transform:
            raw_ = self.transform(raw_)
        if self.transform2:
            proc = self.transform2(proc)

        sample = {'input': raw_, 'standard': proc}

        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        self.image = image
        h, w = self.image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        self.image = transform.resize(self.image, (new_h, new_w))
        return self.image

class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        self.image = image

        h, w = self.image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        self.image = self.image[top: top + new_h, left: left + new_w]
        return self.image

class ToTensor(object):
    def __call__(self, image):
        self.image = image
        if len(self.image.shape) != 3:
            self.image = torch.from_numpy(np.expand_dims(self.image, axis=0)).float()
        else:
            self.image = torch.from_numpy(self.image).float()
        return self.image
