import lumapi
from tqdm import tqdm
import os
import utils
import torch
from interface import *
import numpy as np

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def train(generator, optimizer, scheduler,numIter,gkernlen,gkernsig):
    ## Parameters
    batch_size_start = 100
    batch_size_end = 100
    batch_size_power = 1
    sigma_start = 0.7
    sigma_end = 0.7
    output_dir = "E:/LAB/Project GLOnet-LumeircalAPI/GLOnet-LumericalAPI/Output"
    noise_dims =256
    noise_amplitude = 1
    ## Parameters End


    generator.train() ## network to be in training state

    effs_mean_history = []
    binarization_history = []
    diversity_history = []
    iter0 = 0

    with tqdm(total=numIter) as t:
        it = 0
        while True:
            it += 1
            iter = it + iter0
            normIter = iter / numIter
            batch_size = int(batch_size_start + (batch_size_end - batch_size_start) * (
                        1 - (1 - normIter) ** batch_size_power))
            sigma = sigma_start + (sigma_end - sigma_start) * normIter
            scheduler.step()
            if iter < 1000:
                binary_amp = int(iter / 100) + 1
            else:
                binary_amp = 10

            if it % 5000 == 0 or it > numIter:
                model_dir = os.path.join(output_dir, 'model', 'iter{}'.format(it + iter0))
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
            if it > numIter:
                return

            # sample  z
            z = sample_z(batch_size, noise_dims , noise_amplitude)

            # generate a batch of iamges
            gen_imgs = generator(z, noise_dims,gkernlen,gkernsig)

            # calculate efficiencies and gradients using EM solver
            wavelength = 1550
            effs, gradients = compute_effs_and_gradients(gen_imgs, wavelength) ## to call Lumerical APIs







def sample_z(batch_size, noise_dims , noise_amplitude):
    '''
    smaple noise vector z
    '''
    return (torch.rand(batch_size, noise_dims).type(Tensor)*2.-1.) * noise_amplitude


def compute_effs_and_gradients(gen_imgs, eng, params):
    '''
    Call Lumerical APIs to finish calculation of EM features and gradient
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
    #imgs = gen_imgs.clone().detach()
    #N = imgs.size(0)
    workingDir = "E:\LAB\Project GLOnet-LumeircalAPI\GLOnet-LumericalAPI"
    hide_fdtd_cad = 0
    sim = Simulation(workingDir, hide_fdtd_cad)
    based_script_string = load_from_lsf("grating.lsf")
    based_script = BaseScript(based_script_string)  ## initialize based_script with string
    based_script(sim.fdtd)  ## __call__ to create CAD
    CreatePixels(gen_imgs, sim)
    sim.remove_data_and_save()
    # img = matlab.double(imgs.cpu().numpy().tolist())
    # wavelength = matlab.double([params.wavelength] * N)
    # desired_angle = matlab.double([params.angle] * N)
    #
    # # call matlab function to compute efficiencies and gradients
    #
    # effs_and_gradients = eng.GradientFromSolver_1D_parallel(img, wavelength, desired_angle)
    # effs_and_gradients = Tensor(effs_and_gradients)
    # effs = effs_and_gradients[:, 0]
    # gradients = effs_and_gradients[:, 1:].unsqueeze(1)
    effs = 0
    gradients = 0

    return (effs, gradients)

def CreatePixels(based_matrix,based_simulation):
    lumapi.putMatrix(based_simulation.fdtd.handle, "image_matrix" , based_matrix)
    lumapi.putMatrix(based_simulation.fdtd.handle, "image_shape" , based_matrix.shape)
    based_simulation.fdtd.eval( "row = image_shape(2); \
                                 col = image_shape(1); \
                                 image_matrix; \
                                 create_silica_image(image_matrix,row);")

image_matrix = np.random.randint(0, 2, (1, 256))*2-1
compute_effs_and_gradients(image_matrix,1,1)