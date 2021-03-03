import os
import numpy as np
from net import Generator
import torch
from train import train


if __name__ == '__main__':
    ## Parameters
    cuda = torch.cuda.is_available()
    noise_dims = 256
    gkernlen = 19
    gkernsig = 6
    lr = 1e-03
    beta1 = 0.9
    beta2 = 0.99
    step_size = 5000000
    gamma = 1.0
    numIter = 1000
    ## Parameters End

    generator = Generator(noise_dims,gkernlen,gkernsig)
    if (cuda):
        generator.cuda()

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    ## TODO: restore from

    train(generator, optimizer, scheduler,numIter,gkernlen,gkernsig)
