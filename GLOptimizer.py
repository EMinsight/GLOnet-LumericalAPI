""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.optimize as spo

from lumopt.optimizers.minimizer import Minimizer


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

    def run(self):
        print('GLOnet can Start Here!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Running scipy optimizer')
        print('bounds = {}'.format(self.bounds))
        print('start = {}'.format(self.start_point))
        res = spo.minimize(fun=self.callable_fom,
                           x0=self.start_point,
                           jac=self.callable_jac,
                           bounds=self.bounds,
                           callback=self.callback,
                           options={'maxiter': self.max_iter, 'disp': True, 'gtol': self.pgtol, 'ftol': self.ftol},
                           method=self.method)
        res.x /= self.scaling_factor
        res.fun = -res.fun
        if hasattr(res, 'jac'):
            res.jac = -res.jac * self.scaling_factor
        print('Number of FOM evaluations: {}'.format(res.nit))
        print('FINAL FOM = {}'.format(res.fun))
        print('FINAL PARAMETERS = {}'.format(res.x))
        return res

    def concurrent_adjoint_solves(self):
        return self.method in ['L-BFGS-B', 'BFGS']

