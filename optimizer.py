""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
import copy
import inspect
import numpy as np
import scipy as sp
import scipy.misc

class Optimizer(object):
    """ Base class (or super class) for all optimizers. """

    def __init__(self, max_iter, scaling_factor = 1.0, scale_initial_gradient_to = 0, penalty_fun = None, penalty_jac = None, logging_path = None):
        """ Most optimizers assume the variables to optimize are roughly of order of magnitude of one. Since geometry 
            parameters are usually of the order 1e-9 to 1e-6, it can be useful to scale them to have a magnitude close to one.
            
            Parameters
            ----------
            :param max_iter:       maximum number of iterations.
            :param scaling_factor: scalar or vector of the same length as the optimization parameters; typically used to 
                                   scale the optimization parameters so that they have magnitudes in the range zero to one.
            :param scale_initial_gradient_to: enforces a rescaling of the gradient to change the parameters by at least 
                                              this much; the default value of 0 disables automatic scaling.
            :param: penalty_fun:   penalty function to be added to the figure of merit; it must be a function that takes a vector with
                                   the optimization parameters and returns a single value.
            :param: penalty_jac:   gradient of the penalty function; must be a function that takes a vector with the optimization parameters
                                   and returns a vector of the same length.
            :param logging_path:   directory where the log file should be written to. Default is None which means that it
                                   is written to the current directory.
        """

        self.max_iter = max_iter
        self.scaling_factor = np.array(scaling_factor).flatten()
        self.scale_initial_gradient_to = float(scale_initial_gradient_to)
        self.penalty_fun = penalty_fun if penalty_fun is not None else lambda params: np.zeros(1)
        self.penalty_jac = penalty_jac if penalty_jac is not None else Optimizer.create_jac_approx(self.penalty_fun)
        self.penalty_jac_approx = Optimizer.create_jac_approx(self.penalty_fun)
        self.logging_path = logging_path

        self.logfile = os.path.join(self.logging_path,'optimization_report.txt') if self.logging_path is not None else 'optimization_report.txt'

        if inspect.isfunction(self.penalty_fun):
            bound_args = inspect.signature(self.penalty_fun).bind('params')
            if bound_args.args != ('params',):
                raise UserWarning("penalty function must take one positional argument.")
        else:
            raise UserWarning("penalty function must by a Python function.")
        if inspect.isfunction(self.penalty_jac):
            bound_args = inspect.signature(self.penalty_jac).bind('params')
            if bound_args.args != ('params',):
                raise UserWarning("penalty function gradient must take one positional argument.")
        else:
            raise UserWarning("penalty function gradient must by a Python function.")

        self.current_fom = None
        self.current_gradients = []
        self.current_params = []
        self.fom_hist = []
        self.gradients_hist = []
        self.params_hist = []
        self.iteration = 0
        self.fom_scaling_factor=1
        self.fom_calls = 0

    def initialize(self, start_params, callable_fom, callable_jac, bounds, plotting_function):
        """ Loads the scaled starting point, the bounds and the callables to be used in the optimizer."""

#        assert bounds.shape[0] == start_params.size and bounds.shape[1] == 2
        assert self.scaling_factor.size == 1 or self.scaling_factor.size == start_params.size
        self.define_callback(plotting_function if plotting_function is not None else lambda params: None)
        self.callable_fom, self.callable_jac = self.define_callables(callable_fom, callable_jac)
        self.bounds = bounds
        if self.bounds is not None:
            self.bounds = np.array(self.bounds)
            if len(self.bounds.shape) < 2 or self.bounds.shape[0] != start_params.size or self.bounds.shape[1] != 2:
                raise UserWarning("there must be two bounds for each optimization parameter.")
            for bound in self.bounds:
                if bound[1] - bound[0] <= 0.0:
                    raise UserWarning("bound ranges must be positive.")
            self.bounds *= self.scaling_factor.reshape((self.scaling_factor.size, 1))
        self.reset_start_params(start_params, self.scale_initial_gradient_to)

    def reset_start_params(self, start_params, scale_initial_gradient_to):
        self.scale_initial_gradient_to = scale_initial_gradient_to
        self.start_point = start_params * self.scaling_factor
        if scale_initial_gradient_to != 0.0:
            if self.bounds is None:
                raise UserWarning("bounds are required to scale the initial gradient.")
            self.auto_detect_scaling(scale_initial_gradient_to)
        else:
            self.fom_scaling_factor = 1

    def auto_detect_scaling(self, min_required_rel_change):
        # Calculate the actual epsilon change 
        params = self.start_point
        gradients = self.callable_jac(params)
        params2 = (params - gradients)
        bounds_min = np.array([bound[0] for bound in self.bounds])
        bounds_max = np.array([bound[1] for bound in self.bounds])
        clamped_params = np.maximum(bounds_min, (np.minimum(bounds_max, params2)))
        actual_params = params - clamped_params
        max_change = max(abs(actual_params))
        self.fom_scaling_factor = min_required_rel_change / max_change
        print("Scaling factor is {}".format(self.fom_scaling_factor))

    def define_callback(self, plotting_function):
        def callback(*args):
            """ Called at the end of each iteration to record results."""
            self.params_hist.append(copy.copy(self.current_params))
            self.fom_hist.append(self.current_fom)
            if len(self.current_gradients) > 0:
                self.gradients_hist.append(copy.copy(self.current_gradients))
            #plotting_function(args[0]/self.scaling_factor)
            # self.report_writing()
            self.iteration += 1
        self.callback = callback

    def report_writing(self):
        with open(self.logfile,'a') as f:
            f.write('AT ITERATION {}:  FOM = {}\n'.format(self.iteration, np.array2string(self.fom_hist[-1], separator = ', ', max_line_width = 10000)))
            f.write('PARAMETERS = {}\n'.format(np.array2string(self.params_hist[-1]/self.scaling_factor, separator = ', ', max_line_width = 10000)))
            f.write('\n \n')

    def concurrent_adjoint_solves(self):
        return False
    
    @staticmethod
    def create_jac_approx(fom_func):
        def finite_diff_approx(opt_params):
            jac = np.zeros_like(opt_params)
            for index, param in np.ndenumerate(opt_params):
                def single_param_fom_func(val):
                    local_opt_params = np.array(opt_params)
                    local_opt_params[index] = val
                    return fom_func(local_opt_params)
                jac[index] = sp.misc.derivative(func = single_param_fom_func, x0 = param, dx = 1.0e-5, n = 1, order = 3)
            return jac
        return finite_diff_approx