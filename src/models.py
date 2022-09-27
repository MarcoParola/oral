import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class EmbeddingNet(nn.Module):
    def __init__(self, fc1_size, output_size):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 16, 5), nn.PReLU(),
                                     nn.MaxPool2d(5, stride=3),
                                     nn.Conv2d(16, 32, 3), nn.PReLU(),
                                     nn.MaxPool2d(4, stride=2),
                                     nn.Conv2d(32, 64, 3), nn.PReLU(),
                                     nn.MaxPool2d(3, stride=2),
                                     nn.Conv2d(64, 32, 3), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=1))
        
        self.fc = nn.Sequential(nn.Linear(32 * 19 * 19, fc1_size),
                                nn.PReLU(),
                                nn.Linear(fc1_size, fc1_size),
                                nn.PReLU(),
                                nn.Linear(fc1_size, output_size))
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)



class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)



if __name__ == '__main__':
    net = EmbeddingNet(64, 2)

    triple_net = TripletNet(net)

    summary(net, (3, 300, 300))
    summary(triple_net, [(3, 300, 300), (3, 300, 300), (3, 300, 300)])