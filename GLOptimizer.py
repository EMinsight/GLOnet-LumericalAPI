""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
import logging
import argparse
import numpy as np
from train_and_evaluate import train,evaluate
from net import Generator
import utils
import torch

from  minimizer import Minimizer


class GLOptimizer(Minimizer):
    """ Wrapper for the optimizers in SciPy's optimize package:

            https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize

        Some of the optimization algorithms available in the optimize package ('L-BFGS-G' in particular) can approximate the Hessian from the
        different optimization steps (also called Quasi-Newton Optimization). While this is very powerfull, the figure of merit gradient calculated
        from a simulation using a continuous adjoint method can be noisy. This can point Quasi-Newton methods in the wrong direction, so use them
        with caution.

        Parameters
        ----------
        :param max_iter:       maximum number of iterations; each iteration can make multiple figure of merit and gradient evaluations.
        :param method:         string with the chosen minimization algorithm.
        :param scaling_factor: scalar or a vector of the same length as the optimization parameters; typically used to scale the optimization
                               parameters so that they have magnitudes in the range zero to one.
        :param pgtol:          projected gradient tolerance paramter 'gtol' (see 'BFGS' or 'L-BFGS-G' documentation).
        :param ftol:           tolerance paramter 'ftol' which allows to stop optimization when changes in the FOM are less than this
        :param scale_initial_gradient_to: enforces a rescaling of the gradient to change the optimization parameters by at least this much;
                                          the default value of zero disables automatic scaling.
        :param: penalty_fun:   penalty function to be added to the figure of merit; it must be a function that takes a vector with the
                               optimization parameters and returns a single value.
        :param: penalty_jac:   gradient of the penalty function; must be a function that takes a vector with the optimization parameters
                               and returns a vector of the same length.
    """

    def __init__(self, max_iter, method='L-BFGS-B', scaling_factor=1.0, pgtol=1.0e-5, ftol=1.0e-12,
                 scale_initial_gradient_to=0, penalty_fun=None, penalty_jac=None):
        super(GLOptimizer, self).__init__(max_iter=max_iter,
                                              scaling_factor=scaling_factor,
                                              scale_initial_gradient_to=scale_initial_gradient_to,
                                              penalty_fun=penalty_fun,
                                              penalty_jac=penalty_jac)
        self.method = str(method)
        self.pgtol = float(pgtol)
        self.ftol = float(ftol)

    def run(self): ## train the network with jac

        print('GLOnet can Start Here!!!!!!!!!!!!!!!!!!!!!!!!!!')
        output_dir = 'E:\\LAB\\Project GLOnet-LumeircalAPI\\GLOnet-LumericalAPI\\results'
        restore_from = None

        json_path = os.path.join(output_dir, 'Params.json')
        assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
        params = utils.Params(json_path)

        params.output_dir = output_dir
        params.cuda = torch.cuda.is_available()
        params.restore_from = restore_from
        params.numIter = int(params.numIter)
        params.noise_dims = int(params.noise_dims)
        params.gkernlen = int(params.gkernlen)
        params.step_size = int(params.step_size)

        # make directory
        os.makedirs(output_dir + '/outputs', exist_ok=True)
        os.makedirs(output_dir + '/model', exist_ok=True)
        os.makedirs(output_dir + '/figures/histogram', exist_ok=True)
        os.makedirs(output_dir + '/figures/deviceSamples', exist_ok=True)

        generator = Generator(params)
        if params.cuda:
            generator.cuda()

        # Define the optimizer
        optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))

        # Define the scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)

        # Load model data
        if restore_from is not None:
            params.checkpoint = utils.load_checkpoint(restore_from, generator, optimizer, scheduler)
            logging.info('Model data loaded')

        # Train the model and save
        if params.numIter != 0:
            logging.info('Start training')
            train(generator, optimizer, scheduler, params, func = self.callable_fom , jac = self.callable_jac, callback = self.callback)

        # Generate images and save
        logging.info('Start generating devices')
        #evaluate(generator, eng, numImgs=500, params=params)


        # print('bounds = {}'.format(self.bounds))
        # print('start = {}'.format(self.start_point))
        # res = spo.minimize(fun=self.callable_fom,
        #                    x0=self.start_point,
        #                    jac=self.callable_jac,
        #                    bounds=self.bounds,
        #                    callback=self.callback,
        #                    options={'maxiter': self.max_iter, 'disp': True, 'gtol': self.pgtol, 'ftol': self.ftol},
        #                    method=self.method)
        # res.x /= self.scaling_factor
        # res.fun = -res.fun
        # if hasattr(res, 'jac'):
        #     res.jac = -res.jac * self.scaling_factor
        # print('Number of FOM evaluations: {}'.format(res.nit))
        # print('FINAL FOM = {}'.format(res.fun))
        # print('FINAL PARAMETERS = {}'.format(res.x))
        return

