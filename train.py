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


@hydra.main(config_path="./config/", config_name="config")
def train(cfg):

    # load data
    train_set, validation_set, test_set = get_datasets(cfg)
    triplet_train_loader = DataLoader(train_set, batch_size=cfg.training.batch, shuffle=True)
    triplet_val_loader = DataLoader(validation_set, batch_size=cfg.training.batch, shuffle=True)
    triplet_test_loader = DataLoader(test_set, batch_size=cfg.training.batch, shuffle=True)
    
    print(train_set.__len__(), Counter(train_set.labels))
    print(validation_set.__len__(), Counter(validation_set.labels))
    print(test_set.__len__(), Counter(test_set.labels))
    
    # declare model
    net = EmbeddingNet(cfg.models)
    triple_net = TripletNet(net)

    margin = 1.
    cuda = False
    #loss_fn = TripletLoss(margin)
    loss_fn = TripletMarginLoss(margin=1.0, p=2)
    lr = cfg.training.lr
    optimizer = optim.Adam(triple_net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = cfg.training.epochs
    log_interval = 100


    
    # train
    fit(triplet_train_loader, triplet_val_loader, triple_net, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    # save model
    model_path = os.path.join(cfg.project_path, cfg.models.path, 'triple_net_weights2.pth')
    torch.save(triple_net.state_dict(), model_path)
    









def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target
        loss_outputs = loss_fn(*loss_inputs[:-1])
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs[:-1])
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


if __name__ == '__main__':
    train()