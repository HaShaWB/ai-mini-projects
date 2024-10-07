import argparse
import datetime
from glob import glob
import os

from omegaconf import OmegaConf as OC
from pytorch_lightning import Trainer
import torch

from lightning_model import COIN


def train(args):
    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    model = COIN(hparams)
    ckpt = torch.load(sorted(glob(
        os.path.join(hparams.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt')))[-1],
                      map_location='cpu')
    if args.state_dict_ckeckpoint:
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt['state_dict'])
    trainer = Trainer(
    )
    trainer.test(model,
                 ckpt_path=sorted(glob(
                     os.path.join(hparams.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt')))[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type=int,
                        required=True, help="Resume Checkpoint epoch number")
    parser.add_argument('-s', '--state_dict_ckeckpoint', action='store_true',
                        help="Resume Checkpoint is state dict or not")
    args = parser.parse_args()
    train(args)
