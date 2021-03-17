""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

from optimizer import Optimizer
import numpy as np

class Minimizer(Optimizer):
    """ Base class (or super class) for all optimizers coded as minimizers. """

    def define_callables(self, callable_fom, callable_jac):
        """ Defines the functions that the Optimizer class will use to evaluate the figure of merit and its gradient. The sign
            of the figure of merit and its gradient are flipped here so that the Optimizer class sees a maximizer rather
            than a minimizer.

            Parameters
            ----------
            :param callable_fom: function taking a numpy vector of optimization parameters and returning a scalar figure of merit.
            :param callable_jac: function taking a numpy vector of optimization parameters and returning a vector of the same
                                 size with the computed gradients.
        """

        def callable_fom_local(params):
            params_over_scaling_factor = params / self.scaling_factor
            fom = callable_fom(params_over_scaling_factor)
            fom_penalty = self.penalty_fun(params_over_scaling_factor)
            # print('Actual fom: {}, Penalty term: {}, Total fom: {}'.format(fom, fom_penalty,(fom + fom_penalty)))
            self.current_fom = -(fom + fom_penalty)
            self.current_params = params
            self.fom_calls += 1
            return self.current_fom * self.fom_scaling_factor

        def callable_jac_local(params):
            params_over_scaling_factor = params / self.scaling_factor
            fom_gradients = callable_jac(params_over_scaling_factor) / self.scaling_factor
            fom_penalty_gradients = self.penalty_jac(params_over_scaling_factor) / self.scaling_factor

            ## Use finite difference to check correctness of the user-provided gradient of a penalty
            ## function. Very helpful for debugging. TODO: Should be available to users via a flag!
            # test_grad = True
            # if test_grad:
            #     fom_penalty_gradients_approx = self.penalty_jac_approx(params_over_scaling_factor) / self.scaling_factor
            #     fom_penalty_gradients_approx = np.reshape(fom_penalty_gradients_approx, fom_penalty_gradients.shape)
            #     diff = fom_penalty_gradients-fom_penalty_gradients_approx
            #     filter_h= 5
            #     sub_diff = diff[filter_h:-filter_h,filter_h:-filter_h]

            self.current_gradients = -(fom_gradients + fom_penalty_gradients.flatten())

            # if self.fom_calls==1:
            #     self.callback(params)
            #     print("Finish Call Back!!!!")
            return self.current_gradients * self.fom_scaling_factor

        return callable_fom_local,callable_jac_local