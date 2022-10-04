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



class TripletNet(LightningModule):
    def __init__(self, embedding_net, lr=0.001):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.lr = lr

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def training_step(self, batch, batch_nb):
        (x1, x2, x3), _ = batch
        h1, h2, h3 = self(x1, x2, x3)

        loss = self.loss(h1, h2, h3)
        self.log("train/loss", loss.item())

        return loss

    def get_outputsize(self):
        return self.embedding_net.get_outputsize()

    def get_embedding(self, x):
        return self.embedding_net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    net = EmbeddingNet({'path': 'models', 'dropout': 0.15, 'output_size': 2, 'fc1_size': 64})
    triple_net = TripletNet(net)

    '''
    summary(net, (3, 300, 300))
    summary(triple_net, [(3, 300, 300), (3, 300, 300), (3, 300, 300)]) 
    '''
    img = torch.randn(2, 3, 300, 300)
    triple_net(img, img, img)