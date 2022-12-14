import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import torchvision.models as models
from pytorch_lightning import LightningModule

class EmbeddingNet(nn.Module):

    def __init__(self, hyper_param_dict):
        super(EmbeddingNet, self).__init__()
        fc1_size = hyper_param_dict['fc1_size']
        self.output_size = hyper_param_dict['output_size']
        dropout_rate = hyper_param_dict['dropout']

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, 5), nn.PReLU(),
            nn.Dropout(p=dropout_rate), nn.MaxPool2d(5, stride=3),
            nn.Conv2d(16, 32, 3), nn.PReLU(),
            nn.Dropout(p=dropout_rate), nn.MaxPool2d(4, stride=2),
            nn.Conv2d(32, 64, 3), nn.PReLU(),
            nn.Dropout(p=dropout_rate), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 32, 3), nn.PReLU(),
            nn.Dropout(p=dropout_rate), nn.MaxPool2d(2, stride=1))

        #self.convnet = models.vgg16(pretrained=True)
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 19 * 19, fc1_size),
            nn.PReLU(), nn.Dropout(p=dropout_rate), 
            nn.Linear(fc1_size, fc1_size),
            nn.PReLU(), nn.Dropout(p=dropout_rate), 
            nn.Linear(fc1_size, self.output_size))

        '''
        num_ftrs = self.convnet.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, fc1_size),
            nn.PReLU(), nn.Dropout(p=dropout_rate), 
            nn.Linear(fc1_size, fc1_size),
            nn.PReLU(), nn.Dropout(p=dropout_rate), 
            nn.Linear(fc1_size, self.output_size))
        '''
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_outputsize(self):
        return self.output_size

    def get_embedding(self, x):
        return self.forward(x)

'''

anchor_embeddings = [A1, A2, A3]
positive_embeddings = [P1, P2, P3]
negative_embeddings = [N1, N2, N3]


   A1 A2 A3 P1 P2 P3 N1 N2 N3
A1 1  1  1  1  1  1  0  0  0
A2 1  1  1  1  1  1  0  0  0
A3 1  1  1  1  1  1  0  0  0
P1 1  1  1  1  1  1  0  0  0
P2 1  1  1  1  1  1  0  0  0
P3 1  1  1  1  1  1  0  0  0
N1 0  0  0  0  0  0  1  1  1
N2 0  0  0  0  0  0  1  1  1
N3 0  0  0  0  0  0  1  1  1

'''


class TripletNet(LightningModule):
    def __init__(self, embedding_args, lr=0.001):
        super(TripletNet, self).__init__()
        self.embedding_net = EmbeddingNet(embedding_args)
        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
        self.lr = lr

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def training_step(self, batch, batch_nb):
        (x1, x2, x3), _ = batch
        h1, h2, h3 = self(x1, x2, x3)
        h = torch.cat([h1, h2, h3], dim=0)
        h = F.normalize(h, dim=1)
        contrastive_matrix = torch.matmul(h, h.t()).clip(0, 1)
        
        target_matrix = torch.ones_like(contrastive_matrix)
        target_matrix[h1.shape[0]*2:, :h1.shape[0]*2] = 0
        target_matrix[:h1.shape[0]*2, h1.shape[0]*2:] = 0
        
        loss = F.binary_cross_entropy(contrastive_matrix, target_matrix)

        self.log("train/loss", loss.item())

        return loss

    def get_outputsize(self):
        return self.embedding_net.get_outputsize()

    def get_embedding(self, x):
        return self.embedding_net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    triple_net = TripletNet({'path': 'models', 'dropout': 0.15, 'output_size': 8, 'fc1_size': 64})

    '''
    summary(net, (3, 300, 300))
    summary(triple_net, [(3, 300, 300), (3, 300, 300), (3, 300, 300)]) 
    '''
    img = torch.randn(2, 3, 300, 300)
    triple_net.training_step(((img, img, img), "cls"), 0)