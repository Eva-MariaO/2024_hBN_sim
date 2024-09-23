#shebang?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abtem
import ase
import dask
import os
import sys
from ase.io import read, write

#paths
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_dir = os.path.abspath(os.path.join(parent_dir, 'data', 'data_DFT'))#choose path for results
input_data_dir = os.path.abspath(os.path.join(data_dir, 'input_trajectories'))#choose path of input (.traj)

#import homemade modules by adding hBN_sim directory to sys.path
sys.path.insert(0, parent_dir)
from helper_funcs import timeformat, built_structure, defect_indices, structure_manipulation

#Parameters & Settings
#----------------------------------------------------------------------------------------------------
#setting configuration
abtem.config.set({"device": "gpu", "fft": "fftw"})
dask.config.set({"num_workers": 1})

#FIX
lattice_constant = 2.504 # https://www.sciencedirect.com/science/article/abs/pii/S0025540815300088
size_x = 20 #size of structure
size_y = 13
energy_probe = 60e3
defocus = 0 # 0 or f.e.: 'scherzer'
semiangle_cutoff = 30 
Cs = -10e-6*1e10 #spherical aberration; 10 micrometers

#abberrations: [default: Cs = -10e-6*1e10; astig = 10; coma = 1000]
astig = 0 #f.e. [0, 15, 25] aberration C12, ~few nanometer, input in Å
astig_angle = 0 #f.e. [-0.3, 0, 0.3] #rad
coma = 0 #f.e. [0, 1000, 1500] aberration C21, ~few hundred nanometers, input in Å
aberration_coefficients = {'C10': defocus, 'C30': Cs, 'C12': astig, 'C21': coma}

#import preparation
data_names = [entry for entry in os.listdir(input_data_dir) if entry.endswith('.traj')]

#----------------------------------------------------------------------------------------------------

#simulation function

def simulation(path_or, filename_res):

    #import data from .traj file (time-development trajectory of structure during DFT relaxation)
    atoms = read(path_or ,index=':')   
    hBN_manipulated = atoms[-1] #-1 to use final status of structure after relaxation
        
    #abtem setup: phonons, potential, probe
    frozen_phonons = abtem.FrozenPhonons(hBN_manipulated, num_configs = 20, sigmas = 0.1)
    potential = abtem.Potential(frozen_phonons, sampling = 0.05)
    probe = abtem.Probe(energy = energy_probe, semiangle_cutoff = semiangle_cutoff, 
                        aberrations = aberration_coefficients, astigmatism_angle = astig_angle)
    probe.grid.match(potential)
    sampling = probe.aperture.nyquist_sampling

    #abtem scan & detect
    gridscan = abtem.GridScan(start = [0, 0], end = [10/10, 10/10], fractional = True,
                     potential = potential, sampling = sampling) 
    detector_maadf = abtem.AnnularDetector(inner = 60, outer = 200)
    measurements_total = probe.scan(potential, scan = gridscan, detectors = detector_maadf)

    #export data
    measurements_total.to_zarr(f'{data_dir}/{filename_res}_RAW.zarr')

    
#----------------------------------------------------------------------------------------------------

#run code

if __name__ == "__main__":
    count = 0
    try: 
        #run simulation for all variations
        for data in data_names:
            
            #path and name assignments
            path_origin = (f'{input_data_dir}/{data}')       
            timestamp = timeformat()
            filename_result = (f'{timestamp}_{data}')
            print(filename_result)
            
            #run sim
            simulation(path_origin, filename_result)
            count += 1
            print(count)
                    
        
    except Exception as e:
        print(f"Oups - something went wrong: {e}")
