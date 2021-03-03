###########################################################################################################
## File Name: interface.py
## Descript: Lumerical API related functions.
## Author @ Zhenyu ZHAO 2021/02/21
###########################################################################################################

import lumapi
import os
from inspect import signature
from lumapi import FDTD
from lumapi import MODE

## Reference: lumopt @ https://github.com/chriskeraly/lumopt
class Simulation(object):
    """
        Object to manage the FDTD CAD. 
        Parameters
        ----------
        :param workingDir:    working directory for the CAD session.
        :param hide_fdtd_cad: if true, runs the FDTD CAD in the background.
    """

    def __init__(self, workingDir, hide_fdtd_cad):
        """ Launches FDTD CAD and stores a handle. """
        self.fdtd =  lumapi.FDTD(hide = hide_fdtd_cad)
        self.workingDir = workingDir
        self.fdtd.cd(self.workingDir)

    def run(self, name, iter):
        """ Saves simulation file and runs the simulation. """
        self.fdtd.cd(self.workingDir)
        self.fdtd.save('{}_{}'.format(name,iter))
        self.fdtd.run()

    def remove_data_and_save(self):
        self.fdtd.switchtolayout()
        self.fdtd.save()
        
    def __del__(self):
        self.fdtd.close()

## Get Script Reference: lumopt @ https://github.com/chriskeraly/lumopt
def load_from_lsf(script_file_name):
    """ 
       Loads the provided scritp as a string and strips out all comments. 

       Parameters
       ----------
       :param script_file_name: string specifying a file name.
    """

    with open(script_file_name, 'r') as text_file:
        lines = [line.strip().split(sep = '#', maxsplit = 1)[0] for line in text_file.readlines()]
    script = ''.join(lines)
    if not script:
        raise UserWarning('empty script.')
    return script

## Run script to create a base simulation .fsp file Reference: lumopt @ https://github.com/chriskeraly/lumopt
class BaseScript(object):
    """ 
        Proxy class for creating a base simulation. It acts as an interface to place the appropriate call in the FDTD CAD
        to build the base simulation depending on the input object. Options are:
            1) a Python callable,
            2) any visible *.fsp project file,
            3) any visible *.lsf script file or
            4) a plain string with a Lumerical script.
        
        Parameters:
        -----------
        :script_obj: executable, file name or plain string.
    """

    def __init__(self, script_obj):
        if callable(script_obj):
            self.callable_obj = script_obj
            params = signature(script_obj).parameters
            if len(params) > 1:
                raise UserWarning('function to create base simulation must take a single argument (handle to FDTD CAD).')
        elif isinstance(script_obj, str):
            if '.fsp' in script_obj and os.path.isfile(script_obj) or '.lms' in script_obj and os.path.isfile(script_obj):
                self.project_file = os.path.abspath(script_obj)
            elif '.lsf' in script_obj and os.path.isfile(script_obj):
                self.script_str = load_from_lsf(os.path.abspath(script_obj))
            else:
                self.script_str = str(script_obj)
        else:
            raise UserWarning('object for generating base simulation must be a Python function, a file name or a string with a Lumerical script.')

    def __call__(self, cad_handle):
        return self.eval(cad_handle)

    def eval(self, cad_handle):
        if not isinstance(cad_handle, FDTD) and not isinstance(cad_handle, MODE):
            raise UserWarning('input must be handle returned by lumapi.FDTD.')
        if hasattr(self, 'callable_obj'):
            return self.callable_obj(cad_handle)
        elif hasattr(self, 'project_file'):
            return cad_handle.load(self.project_file)
        elif hasattr(self, 'script_str'):
            return cad_handle.eval(self.script_str)
        else:
            raise RuntimeError('un-initialized object.')

