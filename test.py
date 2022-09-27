from src.datasets import get_dataLoader
import hydra
import os
import torch
import numpy as np
from src.models import EmbeddingNet, TripletNet
from src.losses import TripletLoss
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

ulcer_classes = ['0', '1', '2']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']



@hydra.main(config_path="./config/", config_name="config")
def test(cfg):

    triplet_train_loader, triplet_val_loader, triplet_test_loader = get_dataLoader(cfg)
    cuda = False 
    n_classes = cfg.datasets.n_classes

    # load model
    net = EmbeddingNet(64, 2)
    triple_net = TripletNet(net)
    model_path = os.path.join(cfg.project_path, cfg.models.path, 'triple_net_weights.pth')
    triple_net.load_state_dict(torch.load(model_path)
    triple_net.eval()

    train_embeddings_tl, train_labels_tl = extract_embeddings(triplet_train_loader, triple_net, cuda)
    print(train_embeddings_tl.shape, train_labels_tl.shape)
    plot_embeddings(train_embeddings_tl, train_labels_tl)    
    
    val_embeddings_tl, val_labels_tl = extract_embeddings(triplet_test_loader, triple_net, cuda)
    print(val_embeddings_tl.shape, val_labels_tl.shape)
    plot_embeddings(val_embeddings_tl, val_labels_tl)




def plot_embeddings(embeddings, targets, n_classes, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ulcer_classes)
    plt.show()



def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images[0])] = model.get_embedding(images[0]).data.cpu().numpy()
            labels[k:k+len(images[0])] = target[0].numpy()
            k += len(images)
        
    return embeddings, labels




if __name__ == '__main__':
    test()