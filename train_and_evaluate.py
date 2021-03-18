from tqdm import tqdm
import os
import utils
import torch
import numpy as np
from torchvision.utils import save_image
import scipy.io as io
import logging


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def train(generator, optimizer, scheduler, params ,func,jac,callback,pca=None):
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

            # calculate efficiencies and gradients using Lumerical
            effs, gradients = compute_effs_and_gradients(gen_imgs, params,func,jac,callback)
            # print(effs)
            # print(gradients)

            # free optimizer buffer
            optimizer.zero_grad()

            # construct the loss function
            binary_penalty = params.binary_penalty_start if params.iter < params.binary_step_iter else params.binary_penalty_end
            g_loss = global_loss_function(gen_imgs, effs, gradients, params.sigma, binary_penalty)

            # train the generator
            g_loss.backward()
            optimizer.step()

            # evaluate
            if it % params.plot_iter == 0:
                generator.eval()

                # vilualize generated images at various conditions
                visualize_generated_images(generator, params)

                # evaluate the performance of current generator
                effs_mean, binarization, diversity = evaluate_training_generator(generator,func, params)

                # add to history
                effs_mean_history.append(effs_mean)
                binarization_history.append(binarization)
                diversity_history.append(diversity)

                # plot current history
                utils.plot_loss_history((effs_mean_history, diversity_history, binarization_history), params)
                generator.train()

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


def compute_effs_and_gradients(gen_imgs, params , func , jac , callback):
    '''
    Args:
        imgs: N x C x H
        labels: N x labels_dim
        eng: matlab engine
        params: parameters

    Returns:
        effs: N x 1
        gradients: N x C x H
    '''
    # convert from tensor to numpy array
    imgs = gen_imgs.clone().detach()
    N = imgs.size(0)
    effs = np.zeros((N,1))
    gradients = np.zeros((N,imgs.size(1),imgs.size(2)))
    for i in range(0,N):
        geo_params = imgs.cpu().numpy()[i,:,:].flatten()
        # print("sent params:",geo_params)
        effs[i,:] = -func(geo_params)
        gradients[i,:,:] = jac(geo_params)
        callback()
    effs = Tensor(effs)
    gradients = Tensor(gradients)
    return (effs, gradients)


def visualize_generated_images(generator, params, n_row=4, n_col=4):
    # generate images and save
    fig_path = params.output_dir + '/figures/deviceSamples/Iter{}.png'.format(params.iter)

    z = sample_z(n_col * n_row, params)
    imgs = generator(z, params)
    imgs_2D = imgs.unsqueeze(2).repeat(1, 1, 64, 1)
    save_image(imgs_2D, fig_path, n_row, range=(-1, 1))


def evaluate_training_generator(generator, func,params, num_imgs=10):
    # generate images
    z = sample_z(num_imgs, params)
    imgs = generator(z, params)

    # efficiencies of generated images
    effs = compute_effs(imgs, func,params)
    effs_mean = torch.mean(effs.view(-1))

    # binarization of generated images
    binarization = torch.mean(torch.abs(imgs.view(-1))).cpu().detach().numpy()

    # diversity of generated images
    diversity = torch.mean(torch.std(imgs, dim=0)).cpu().detach().numpy()

    # plot histogram
    fig_path = params.output_dir + '/figures/histogram/Iter{}.png'.format(params.iter)
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.iter, fig_path)

    return effs_mean, binarization, diversity


def compute_effs(imgs, func , params):
    '''
    Args:
        imgs: N x C x H
        eng: matlab engine
        params: parameters

    Returns:
        effs: N x 1
    '''
    imgs = imgs.clone().detach()
    N = imgs.size(0)
    effs = np.zeros((N, 1))
    for i in range(0, N):
        geo_params = imgs.cpu().numpy()[i, :, :].flatten()
        # print("sent params:",geo_params)
        effs[i, :] = func(geo_params)

    return Tensor(effs)


def evaluate(generator, func, numImgs, params):
    generator.eval()

    # generate images
    z = sample_z(numImgs, params)
    images = generator(z, params)
    logging.info('Generation is done. \n')

    # evaluate efficiencies
    images = torch.sign(images)
    effs = compute_effs(images, func, params)

    # save images
    filename = 'imgs_w' + str(params.wavelength) + '_a' + str(params.angle) + 'deg.mat'
    file_path = os.path.join(params.output_dir, 'outputs', filename)
    io.savemat(file_path, mdict={'imgs': images.cpu().detach().numpy(),
                                 'effs': effs.cpu().detach().numpy()})

    # plot histogram
    fig_path = params.output_dir + '/figures/Efficiency.png'
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.numIter, fig_path)