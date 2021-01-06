import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


class DeeplabVGG(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.layer1 = nn.Sequential(*([features[i] for i in range(0,16)]))
        self.layer2 = nn.Sequential(*([features[i] for i in range(16,21)]))
        self.layer3 = nn.Sequential(*([features[i] for i in range(21,len(features))]))
        self.layer4 = nn.Sequential(*([fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.tanh = nn.Tanh()

        self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)

    def forward(self, x, D, domain):
        x1 = self.layer1(x)        
        x2 = self.layer2(x1)        
        x3 = self.layer3(x2) 
        if domain=='source':
        	x3_a3 = x3
        if domain=='target':
        	a3 = D[0](x3)
        	a3 = self.tanh(a3)
        	a3 = torch.abs(a3)
        	a3_big = a3.expand(x3.size())
        	x3_a3 = a3_big*x3 + x3    
        x4 = self.layer4(x3_a3)        
        x5 = self.classifier(x4)
        return x3, x5

    def optim_parameters(self, args):
        return self.parameters()

class DeeplabVGG_val(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG_val, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.layer1 = nn.Sequential(*([features[i] for i in range(0,16)]))
        self.layer2 = nn.Sequential(*([features[i] for i in range(16,21)]))
        self.layer3 = nn.Sequential(*([features[i] for i in range(21,len(features))]))
        self.layer4 = nn.Sequential(*([fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.tanh = nn.Tanh()

        self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)

    def forward(self, x, D, domain):
        x1 = self.layer1(x)        
        x2 = self.layer2(x1)        
        x3 = self.layer3(x2) 
        if domain=='source':
        	x3_a3 = x3
        if domain=='target':
        	a3 = D[0](x3)
        	a3 = self.tanh(a3)
        	a3 = torch.abs(a3)
        	a3_big = a3.expand(x3.size())
        	x3_a3 = a3_big*x3 + x3    
        x4 = self.layer4(x3_a3)        
        x5 = self.classifier(x4)
        return x3, x5

    def optim_parameters(self, args):
        return self.parameters()
