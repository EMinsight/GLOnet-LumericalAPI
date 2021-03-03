import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, noise_dims,gkernlen,gkernsig):
        super().__init__()

        self.noise_dim = noise_dims

        self.gkernel = gkern1D(gkernlen, gkernsig)

        self.FC = nn.Sequential(
            nn.Linear(self.noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32*16, bias=False),
            nn.BatchNorm1d(32*16),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 1, 5),
            )


    def forward(self, noise, binary_amp):
        net = self.FC(noise)
        net = net.view(-1, 16, 32)
        net = self.CONV(net)    
        net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        # net = conv1d_meta(net , self.gkernel)
        net = torch.tanh(net* binary_amp) * 1.05
        return net



## For test
# net = Generator(256,19,6)
# print(net)
# torch.save(net, "E:/LAB/Project GLOnet-LumeircalAPI/GLOnet-LumericalAPI/Output/test.pth")