from math import sqrt

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn

import dataloader


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1.):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # init w and b#
        # Initialize layers following SIREN paper
        w_std = (sqrt(6. / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)
        self.w0 = w0

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.w0 * x)
        return x


class COIN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        c = self.hparams.arch.channels
        self.mid = nn.ModuleList(
            [Siren(c, c) for i in range(self.hparams.arch.layers - 2)]
        )
        self.model = nn.Sequential(
            Siren(2, c, self.hparams.arch.init_scale),
            *self.mid,
            Siren(c, 15),
        )
        self.mse = nn.MSELoss()

        # Save Image for tblogger

        self.images = list()

        for image_path in self.hparams.data.images:
            im = Image.open(image_path)
            im = (np.array(im, dtype=np.int32) - 128) / 128.  # normalize rgb [0,255]-> [-1,1]
            im = im[:, :, :3]  # remove alpha channel
            self.images.append(torch.tensor(im))

        self.images = torch.stack(self.images)


        im = Image.open(self.hparams.data.image)
        im = (np.array(im, dtype=np.int32) - 128) / 128.  # normalize rgb [0,255]-> [-1,1]
        im = im[:, :, :3]  # remove alpha channel
        self.image = torch.tensor(im)

    def forward(self, x):
        # [B, 2] --> [B, 3]
        x = self.model(x)
        return x

    def common_step(self, x, rgbs):
        output = self(x)
        # 15차원 출력을 5개의 3차원 RGB로 변환
        output = output.view(-1, 5, 3)
        rgbs = rgbs.view(-1, 5, 3)
        loss = self.mse(output, rgbs)
        return loss, output

    def training_step(self, batch, batch_nb):
        x, rgbs = batch  # coordinate [B,2], rgb [B,3]
        loss, _ = self.common_step(x, rgbs)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, rgbs = batch
        loss, output = self.common_step(x, rgbs)

        # 5개의 이미지를 복원
        output_images = output.view(-1, 5, self.image.shape[0], self.image.shape[1], 3)
        output_images = (128 * output_images + 128).detach().cpu().to(torch.int32).numpy()

        # 각 이미지를 로깅
        for i in range(output_images.shape[1]):
            img = output_images[0, i]  # 첫 번째 배치의 i번째 이미지
            self.logger.log_image(img, self.image, self.current_epoch)

        self.log('val_loss', loss)
        return {'loss': loss, 'output': output}

    def test_step(self, batch, batch_nb):
        x, rgbs = batch  # [B, H*W, 2], [B, H*W, 3]
        loss, output = self.common_step(x, rgbs)
        self.log('test_loss', loss)

        with torch.no_grad():
            # 5개의 이미지를 복원
            output_images = output.view(-1, 5, self.image.shape[0], self.image.shape[1], 3)
            output_images = (128 * output_images + 128).detach().cpu().to(torch.int32).numpy()

            # 각 이미지를 저장
            for i in range(output_images.shape[1]):
                img = output_images[0, i]  # 첫 번째 배치의 i번째 이미지
                im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='RGB')
                im.save(f'./figure/recon_{i}.png', format='png')

        return {'test_loss': loss, 'output': output}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.hparams.train.lr,
                               eps=self.hparams.train.opt_eps,
                               betas=(self.hparams.train.beta1,
                                      self.hparams.train.beta2),
                               weight_decay=self.hparams.train.weight_decay)
        return opt

    def train_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 1)

    def test_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 2)
