from __future__ import print_function
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

"""
All test models are defined here. 
"""

#Based off vgg net
class Net1(nn.Module):
    def __init__(self, num_classes=10):
        super(Net1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride = 2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropoutlayer = nn.Sequential(
            nn.Dropout())
        self.fc = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(10, num_classes)
        self.activation = nn.Sequential( 
            nn.Softmax())
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.dropoutlayer(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        #out = self.layer8(out)
        #out = out.reshape(out.size(0),-2)
        #print(out.shape)
        out = out.reshape(out.size(0),-1)
        #print(out.shape)
        out = self.fc(out)
 #       out = self.dropoutlayer(out)
        out = self.fc2(out)
        out = self.activation(out)
        return out

#Based off vgg net with connections between layers
class Net2(nn.Module):
    def __init__(self, num_classes=10):
        super(Net2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64))
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride = 2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropoutlayer = nn.Sequential(
            nn.Dropout())
        self.fc = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(10, num_classes)
        self.activation = nn.Sequential( 
            nn.Softmax())
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        outpool1 = self.pool1(torch.cat((out3, out4),1))
        out5 = self.layer5(outpool1)
        out6 = self.layer6(out5)
        outpool2 = self.pool2(torch.cat((out5, out6),1))
        out7 = self.layer7(outpool2)
        out = out7.reshape(out7.size(0),-1)
        out = self.fc(out)
        out = self.fc2(out)
        out = self.activation(out)
        return out

#Based off google net
class Net3(nn.Module):
    def __init__(self, num_classes=10):
        super(Net3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=1, stride = 1, padding =1),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3, stride =1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer4_1 = nn.Sequential(
            nn.Conv2d(16,16,kernel_size=1, stride =1, padding=1),
            nn.ReLU(True))
        self.layer4_3 = nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3, stride =1, padding = 1),
            nn.ReLU(True))
        self.layer4_5 = nn.Sequential(
            nn.Conv2d(16,16,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True))
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1, stride =1, padding=1),
            nn.ReLU(True))
        self.layer5_3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride =1, padding = 1),
            nn.ReLU(True))
        self.layer5_5 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True))
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=1, stride =1, padding=1),
            nn.ReLU(True))
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3, stride =1, padding = 1),
            nn.ReLU(True))
        self.layer6_5 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True))
        self.layer7_1 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=1, stride =1, padding=1),
            nn.ReLU(True))
        self.layer7_3 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3, stride =1, padding = 1),
            nn.ReLU(True))
        self.layer7_5 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True))
        self.layermaxpool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc = nn.Linear(1024, num_classes)
        self.activation = nn.Softmax()
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out4_1 = self.layer4_1(out)
        out4_3 = self.layer4_3(out4_1)
        out4_5 = self.layer4_5(out4_1)
        outpool = self.layermaxpool(out)
        out4_1_1 = self.layer4_1(outpool)
        outinception = torch.cat((out4_1, out4_3),1)
        outinception = torch.cat((outinception, out4_5),1)
        outinception = torch.cat((outinception, out4_1_1),1)
        outinception = self.maxpool(outinception)
        out5_1 = self.layer5_1(outinception)
        out5_3 = self.layer5_3(out5_1)
        out5_5 = self.layer5_5(out5_3)
        outpool = self.layermaxpool(outinception)
        out5_1_1= self.layer5_1(outpool)
        outinception2 = torch.cat((out5_1,out5_3),1)
        outinception2 = torch.cat((outinception2,out5_5),1)
        outinception2 = torch.cat((outinception2, out5_1_1),1)
        outinception2 = self.maxpool(outinception2)
        out6_1 = self.layer6_1(outinception2)
        out6_3 = self.layer6_3(out6_1)
        out6_5 = self.layer6_5(out6_3)
        outpool = self.layermaxpool(outinception2)
        out6_1_1= self.layer6_1(outpool)
        outinception3 = torch.cat((out6_1,out6_3),1)
        outinception3 = torch.cat((outinception3,out6_5),1)
        outinception3 = torch.cat((outinception3, out6_1_1),1)
        out = self.maxpool(outinception3)
        #out7_1 = self.layer7_1(outinception3)
        #out7_3 = self.layer7_3(out7_1)
        #out7_5 = self.layer7_5(out7_3)
        #outpool = self.layermaxpool(outinception3)
        #out7_1_1= self.layer7_1(outpool)
        #outinception4 = torch.cat((out7_1,out7_3),1)
        #outinception4 = torch.cat((outinception4,out7_5),1)
        #outinception4 = torch.cat((outinception4, out7_1_1),1)
        #out = self.avgpool(outinception4)
        #out = self.maxpool(outinception4)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        out = self.activation(out)
        return out

class ResNet(ResNet):
   def __init__(self, num_classes=10):
       super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2],
                                         num_classes=num_classes)
       self.conv1 = torch.nn.Conv2d(1, 64,
                                    kernel_size=(7, 7),
                                    stride=(2, 2),
                                    padding=(3, 3), bias=False)
   def forward(self, x):
       return torch.softmax(
           super(MnistResNet, self).forward(x), dim=-1)
