from os import path, makedirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from PIL import Image  # Import Image from PIL

class TensorBoardLoggerExpanded(TensorBoardLogger):
    def __init__(self, hparams):
        super().__init__(hparams.log.tensorboard_dir, name=hparams.name,
                         default_hp_metric=False)
        self.hparams = hparams
        self.log_hyperparams(hparams)

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def plot_to_numpy(self, image, epoch):
        fig = plt.figure(figsize=(5, 4))
        plt.title(f'Epoch {epoch}')
        plt.imshow(np.clip(image, 0, 255),
                   aspect='equal',
                   )
        fig.canvas.draw()
        data = self.fig2np(fig)
        plt.close()
        return data

    def log_image(self, output, image, epoch):
        # output은 이미지를 복원한 numpy 배열로 가정
        output_image = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode='RGB')
        output_image_np = np.array(output_image)

        if epoch == 99:
            image_np = (128 * image + 128).detach().cpu().to(torch.int32).numpy()
            image_np = self.plot_to_numpy(image_np, epoch)
            self.experiment.add_image(path.join(self.save_dir, 'image'),
                                      image_np,
                                      epoch,
                                      dataformats='HWC')

        output_np = self.plot_to_numpy(output_image_np, epoch)
        self.experiment.add_image(path.join(self.save_dir, 'output'),
                                  output_np,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
