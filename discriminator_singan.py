import torch
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    # TODO: Pad whole input to discriminator rather than in each layer
    def __init__(self, nfc=32, min_nfc=32, num_layer=5, nc_im=3, ker_size=3, padd_size=1):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(nfc)
        self.head = ConvBlock(nc_im,N,ker_size,padd_size,1)
        self.body = nn.Sequential()
        for i in range(num_layer-2):
            N = int(nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,min_nfc),max(N,min_nfc),ker_size,padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,min_nfc),1,kernel_size=ker_size,stride=1,padding=padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    def _calc(disc_interpolates):
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    if isinstance(disc_interpolates, list):
        # Average penalties of each discriminator
        penalties = torch.stack([_calc(disc_) for disc_ in disc_interpolates])
        gradient_penalty = penalties.mean()
    else:
        gradient_penalty = _calc(disc_interpolates)
        
    return gradient_penalty
