import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (224,224), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # (224-5)+1 = 221, 221/2 = 110 after pooling
        self.conv2 = nn.Conv2d(32, 64, 4) # (110-4)+1 = 107, 107/2 = 53 after pooling
        self.conv3 = nn.Conv2d(64, 128, 3) # (53-3)+1 = 51, 51/2 = 25 after pooling
        self.conv4 = nn.Conv2d(128, 256, 2) # (25-2)+1 = 24, 24/2 = 12 after pooling
        self.conv5 = nn.Conv2d(256, 512, 1) # (12-1)+1 = 12, 12/2 = 6 after pooling
        
        # Dropout layers
        self.dropout1 = nn.Dropout2d(p=0.1) # for conv2d
        self.dropout2 = nn.Dropout2d(p=0.2) # for conv2d
        self.dropout3 = nn.Dropout2d(p=0.3) # for conv2d
        self.dropout4 = nn.Dropout2d(p=0.4) # for conv2d
        self.dropout5 = nn.Dropout2d(p=0.5) # for conv2d

        self.dropout6 = nn.Dropout(p=0.5) # for linear1d
        self.dropout7 = nn.Dropout(p=0.6) # for linear1d

        # Maxpolling layer
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Dense layers, input_feature = 512*6*6 = 18432
        self.fc1 = nn.Linear(18432,1000)
        self.fc2 = nn.Linear(1000,512)
        self.out = nn.Linear(512,68*2)

        # Batchnorm layers

        self.BF1 = nn.BatchNorm2d(32) # for conv2d
        self.BF2 = nn.BatchNorm2d(64) # for conv2d
        self.BF3 = nn.BatchNorm2d(128) # for conv2d
        self.BF4 = nn.BatchNorm2d(256) # for conv2d

        self.BF5 = nn.BatchNorm1d(18432) # for linear1d
        self.BF6 = nn.BatchNorm1d(1000) # for linear1d
        self.BF7 = nn.BatchNorm1d(512) # for linear1d
        

        
    def forward(self, x):
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.BF1(self.dropout1(self.pool(F.relu(self.conv1(x)))))
        x = self.BF2(self.dropout2(self.pool(F.relu(self.conv2(x)))))
        x = self.BF3(self.dropout3(self.pool(F.relu(self.conv3(x)))))
        x = self.BF4(self.dropout4(self.pool(F.relu(self.conv4(x)))))
        x = self.dropout5(self.pool(F.relu(self.conv5(x))))
        # flatten
        x = self.BF5(x.view(x.size(0),-1))
        x = self.BF6(self.dropout6(F.relu(self.fc1(x))))
        x = self.BF7(self.dropout7(F.relu(self.fc2(x))))
        x = self.out(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
