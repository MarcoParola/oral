from src.datasets import get_datasets
import hydra
import os
import torch
import numpy as np
from src.models import EmbeddingNet, TripletNet
from src.losses import TripletLoss
from torch.nn import TripletMarginLoss
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torchsummary import summary
from test import extract_embeddings
from collections import Counter
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from omegaconf import OmegaConf
import flatdict

def hp_from_cfg(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(flatdict.FlatDict(cfg, delimiter="/"))

@hydra.main(config_path="./config/", config_name="config")
def train(cfg):

    # load data
    train_set, validation_set, test_set = get_datasets(cfg)
    triplet_train_loader = DataLoader(train_set, batch_size=cfg.training.batch, shuffle=True, num_workers=os.cpu_count())
    triplet_val_loader = DataLoader(validation_set, batch_size=cfg.training.batch, shuffle=True, num_workers=os.cpu_count())
    triplet_test_loader = DataLoader(test_set, batch_size=cfg.training.batch, shuffle=True, num_workers=os.cpu_count())
    
    print(train_set.__len__(), Counter(train_set.labels))
    print(validation_set.__len__(), Counter(validation_set.labels))
    print(test_set.__len__(), Counter(test_set.labels))
    
    loggers = list()
    if cfg.training.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)

    # declare model
    triple_net = TripletNet(cfg.models)

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=cfg.training.epochs,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        logger=loggers,
        log_every_n_steps=1
    )

    trainer.fit(triple_net, triplet_train_loader)

    # save model
    os.makedirs(cfg.models.path, exist_ok=True)
    model_path = os.path.join(cfg.project_path, cfg.models.path, 'triple_net_weights2.pth')
    torch.save(triple_net.state_dict(), model_path)
    



if __name__ == '__main__':
    train()