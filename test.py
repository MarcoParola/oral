from src.datasets import get_datasets
import hydra
from hydra.utils import get_original_cwd
import os
import torch
import numpy as np
from src.models import EmbeddingNet, TripletNet
from src.losses import TripletLoss
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn import decomposition
from torch.utils.data import DataLoader
from collections import Counter

ulcer_classes = ['0', '1', '2']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']



@hydra.main(config_path="./config/", config_name="config")
def test(cfg):

    train_set, validation_set, test_set = get_datasets(cfg)
    triplet_train_loader = DataLoader(train_set, batch_size=cfg.training.batch, shuffle=True)
    triplet_val_loader = DataLoader(validation_set, batch_size=cfg.training.batch, shuffle=True)
    triplet_test_loader = DataLoader(test_set, batch_size=cfg.training.batch, shuffle=True)

    print(train_set.__len__(), Counter(train_set.labels))
    print(validation_set.__len__(), Counter(validation_set.labels))
    print(test_set.__len__(), Counter(test_set.labels))

    cuda = torch.cuda.is_available() 
    n_classes = cfg.datasets.n_classes

    # load model
    triple_net = TripletNet(cfg.models)
    model_path = os.path.join(cfg.project_path, cfg.models.path, 'triple_net_weights2.pth')
    triple_net.load_state_dict(torch.load(model_path))
    triple_net.eval()

    pca = decomposition.PCA(n_components=2)
    
    train_embeddings_tl, train_labels_tl = extract_embeddings(triplet_train_loader, triple_net, cuda)
    pca.fit(train_embeddings_tl)
    train_embeddings_tl = pca.transform(train_embeddings_tl)
    print(train_embeddings_tl.shape, train_labels_tl.shape)
    plot_embeddings(train_embeddings_tl, train_labels_tl, n_classes, save="train.png")    
    
    val_embeddings_tl, val_labels_tl = extract_embeddings(triplet_val_loader, triple_net, cuda)
    print(val_embeddings_tl.shape, val_labels_tl.shape)
    val_embeddings_tl = pca.transform(val_embeddings_tl)
    plot_embeddings(val_embeddings_tl, val_labels_tl, n_classes, save="val.png")




def plot_embeddings(embeddings, targets, n_classes, xlim=None, ylim=None, save=None):
    plt.figure(figsize=(10,10))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        print(len(inds), inds)
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.3, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ulcer_classes)
    plt.show()
    if not save is None:
        folder = os.path.join(get_original_cwd(), "plots")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, save))



def extract_embeddings(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        if cuda:
            model.to("cuda")
        embeddings = np.zeros((len(dataloader.dataset), model.get_outputsize()))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = [img.cuda() for img in images]
            embeddings[k:k+len(images[0])] = model.get_embedding(images[0]).data.cpu().numpy()
            labels[k:k+len(images[0])] = target[0].numpy()
            k += len(images)
        
    return embeddings, labels




if __name__ == '__main__':
    test()