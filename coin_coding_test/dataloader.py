from os import path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def create_coin_dataloader(hparams, cv):
    def collate_fn(batch):
        x_list = list()
        rgb_list = list()
        for x, rgb in batch:
            x_list.append(x)
            rgb_list.append(rgb)
        x_list = torch.stack(x_list, dim=0)
        rgb_list = torch.stack(rgb_list, dim=0)

        return x_list, rgb_list

    DS = ImageDataset(hparams, cv)
    if cv == 0:
        return DataLoader(dataset=DS,
                          batch_size=1,  # hparams.train.batch_size,
                          shuffle=True,
                          # collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=DS,
                          batch_size=1,
                          drop_last=False,
                          shuffle=False,
                          )


class ImageDataset(Dataset):
    def __init__(self, hparams, cv=0):  # cv 0: train, 1: val, 2: test
        self.hparams = hparams
        self.cv = cv

        image_paths = hparams.data.images

        self.images = list()



        for image_path in image_paths:
            assert path.isfile(image_path), f"given hparam image path({image_path}) is not a file"
            im = Image.open(image_path)
            im = (np.array(im, dtype=np.int32) - 128) / 128.  # [0,255]-> [-1,1]
            im = im[:, :, :3]  # remove alpha channel
            self.H, self.W, _ = im.shape  # [500, 400]
            im = torch.tensor(im, dtype=torch.float32)
            self.images.append(im)

        h = torch.arange(self.H) / (self.H - 1.) * 2 - 1.  # [-1, 1]
        w = torch.arange(self.W) / (self.W - 1.) * 2 - 1.  # [-1, 1]
        self.x = torch.stack(torch.meshgrid(h, w), dim=-1).view(-1, 2)

        self.rgbs = list()
        for image in self.images:
            self.rgbs.append(image.view(-1, 3).detach())
        self.rgbs = torch.cat(self.rgbs, dim=0)


    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.x, self.rgbs
