######## IMPORTS ########
# General purpose imports
import os
import numpy as np
import scipy as sp

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from GLOptimizer import *
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from GLOptimization import Optimization

######## DEFINE BASE SIMULATION ########
# Use the same script for both simulations, but it's just to keep the example simple. You could use two.
script_1550 = load_from_lsf(os.path.join(os.path.dirname(__file__), 'grating_test.lsf'))

######## DEFINE SPECTRAL RANGE #########
wavelengths_1550 = Wavelengths(start = 1550e-9, stop = 1550e-9, points = 1)

######## DEFINE OPTIMIZABLE GEOMETRY ########

## Definition Start
def make_polygon_func(center_x,center_y,i):
    def make_rectangle(params,n_points = 16):
        wg_width = 0.5e-6
        #print(params)
        width = (params[i] + 1)*0.035e-6
        points_x = np.array([center_x - width/2, center_x + width/2, center_x + width/2 , center_x - width/2])
        points_y = np.array([center_y - wg_width/2 , center_y - wg_width/2 ,  center_y + wg_width/2 , center_y + wg_width/2])
        polygon_points = np.array([(x, y) for x, y in zip(points_x, points_y)])
        return polygon_points
    return make_rectangle

bounds = [(-1.0, 1.0)]*16
initial_params = np.array([0.0]*16)
grating_number = 16
designarealength = 1.12e-6
width_max = 0.07e-6
center_x_start = -designarealength/2 + width_max/2
center_y_start = 0
i = 0
Polygon_Series = FunctionDefinedPolygon(func = make_polygon_func(center_x_start,center_y_start,i), initial_params = initial_params, bounds = bounds, z = 0, depth = 220e-9, eps_out = 2.8 ** 2, eps_in = 1.44 ** 2, edge_precision = 5, dx = 0.1e-9)
for i in range(1,grating_number):
    center_x_start += width_max
    Polygon_Series = Polygon_Series * FunctionDefinedPolygon(func = make_polygon_func(center_x_start,center_y_start,i), initial_params = initial_params, bounds = bounds, z = 0, depth = 220e-9, eps_out = 2.8 ** 2, eps_in = 1.44 ** 2, edge_precision = 5, dx = 0.1e-9)



######## DEFINE FIGURE OF MERIT ########
# Although we are optimizing for the same thing, two separate fom objects must be created.

fom_1550 = ModeMatch(monitor_name = 'fom', mode_number = 1, direction = 'Forward', target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1)

######## DEFINE OPTIMIZATION ALGORITHM ########
#For the optimizer, they should all be set the same, but different objects. Eventually this will be improved
optimizer_1550 = GLOptimizer(max_iter = 10, method = 'L-BFGS-B', scaling_factor = 1, pgtol = 1e-9)

######## PUT EVERYTHING TOGETHER ########
opt_1550 = Optimization(base_script = script_1550, wavelengths = wavelengths_1550, fom = fom_1550, geometry = Polygon_Series, optimizer = optimizer_1550, hide_fdtd_cad = False, use_deps = True)
opt = opt_1550

######## RUN THE OPTIMIZER ########
opt.run()