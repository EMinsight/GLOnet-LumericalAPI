from tqdm import tqdm
import os
import utils
import torch
import numpy as np
import scipy as sp
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def train(generator, optimizer, scheduler, params ,func,jac,pca=None):
    generator.train()

    # initialization
    if params.restore_from is None:
        effs_mean_history = []
        binarization_history = []
        diversity_history = []
        iter0 = 0
    else:
        effs_mean_history = params.checkpoint['effs_mean_history']
        binarization_history = params.checkpoint['binarization_history']
        diversity_history = params.checkpoint['diversity_history']
        iter0 = params.checkpoint['iter']

    # training loop
    with tqdm(total=params.numIter) as t:
        it = 0
        while True:
            it += 1
            params.iter = it + iter0

            # normalized iteration number
            normIter = params.iter / params.numIter

            # specify current batch size
            params.batch_size = int(params.batch_size_start + (params.batch_size_end - params.batch_size_start) * (
                        1 - (1 - normIter) ** params.batch_size_power))

            # sigma decay
            params.sigma = params.sigma_start + (params.sigma_end - params.sigma_start) * normIter

            # learning rate decay
            scheduler.step()

            # binarization amplitude in the tanh function
            if params.iter < 1000:
                params.binary_amp = int(params.iter / 100) + 1
            else:
                params.binary_amp = 10

            # save model
            if it % 5000 == 0 or it > params.numIter:
                model_dir = os.path.join(params.output_dir, 'model', 'iter{}'.format(it + iter0))
                os.makedirs(model_dir, exist_ok=True)
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_state_dict': optimizer.state_dict(),
                                       'scheduler_state_dict': scheduler.state_dict(),
                                       'effs_mean_history': effs_mean_history,
                                       'binarization_history': binarization_history,
                                       'diversity_history': diversity_history
                                       },
                                      checkpoint=model_dir)

            # terminate the loop
            if it > params.numIter:
                return

                # sample  z
            z = sample_z(params.batch_size, params)

            # generate a batch of iamges
            gen_imgs = generator(z, params)
            print("Generated Images !!!!!!")
            print(gen_imgs.shape())

            # # calculate efficiencies and gradients using EM solver
            # effs, gradients = compute_effs_and_gradients(gen_imgs, eng, params)
            #
            # # free optimizer buffer
            # optimizer.zero_grad()
            #
            # # construct the loss function
            # binary_penalty = params.binary_penalty_start if params.iter < params.binary_step_iter else params.binary_penalty_end
            # g_loss = global_loss_function(gen_imgs, effs, gradients, params.sigma, binary_penalty)
            #
            # # train the generator
            # g_loss.backward()
            # optimizer.step()

            # evaluate
            # if it % params.plot_iter == 0:
            #     generator.eval()
            #
            #     # vilualize generated images at various conditions
            #     visualize_generated_images(generator, params)
            #
            #     # evaluate the performance of current generator
            #     effs_mean, binarization, diversity = evaluate_training_generator(generator, eng, params)
            #
            #     # add to history
            #     effs_mean_history.append(effs_mean)
            #     binarization_history.append(binarization)
            #     diversity_history.append(diversity)
            #
            #     # plot current history
            #     utils.plot_loss_history((effs_mean_history, diversity_history, binarization_history), params)
            #     generator.train()

            t.update()


def sample_z(batch_size, params):
    '''
    smaple noise vector z
    '''
    return (torch.rand(batch_size, params.noise_dims).type(Tensor)*2.-1.) * params.noise_amplitude


def global_loss_function(gen_imgs, effs, gradients, sigma=0.5, binary_penalty=0):
    '''
    Args:
        gen_imgs: N x C x H (x W)
        effs: N x 1
        gradients: N x C x H (x W)
        max_effs: N x 1
        sigma: scalar
        binary_penalty: scalar
    '''

    # efficiency loss
    eff_loss_tensor = - gen_imgs * gradients * (1. / sigma) * (torch.exp(effs / sigma)).view(-1, 1, 1)
    eff_loss = torch.sum(torch.mean(eff_loss_tensor, dim=0).view(-1))

    # binarization loss
    binary_loss = - torch.mean(torch.abs(gen_imgs.view(-1)) * (2.0 - torch.abs(gen_imgs.view(-1))))

    # total loss
    loss = eff_loss + binary_loss * binary_penalty

    return loss
