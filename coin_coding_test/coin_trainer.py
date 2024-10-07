import argparse
import datetime
from glob import glob
import os

from omegaconf import OmegaConf as OC
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_model import COIN
from tblogger import TensorBoardLoggerExpanded


def train(args):
    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    model = COIN(hparams)
    tblogger = TensorBoardLoggerExpanded(hparams)
    filename = f'{hparams.log.name}_{now}_{{epoch}}'
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.log.checkpoint_dir,
                                          filename=filename,
                                          verbose=True,
                                          save_last=True,
                                          save_top_k=3,
                                          monitor='val_loss',
                                          mode='min',
                                          prefix='')

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.train.gpus,
        check_val_every_n_epoch=100,
        max_epochs=1000,
        logger=tblogger,
        progress_bar_refresh_rate=4,
        resume_from_checkpoint=None if args.resume_from is None \
            else sorted(glob(
            os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}.ckpt')))[-1],
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type=int,
                        required=False, help="Resume Checkpoint epoch number")
    args = parser.parse_args()
    train(args)
