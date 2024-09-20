#shebang?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abtem
import ase
import dask
import os
from ase.io import read, write

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

#abberrations: [scale: Cs = -10e-6*1e10; astig = 10; coma = 1000]
astig = 0 #[0, 15, 25] #25 aberration C12, ~few nanometers
astig_angle = 0 #[-0.3, 0, 0.3] #+/-0.3 #rad
coma = 0 #[0, 1000, 1500] #1000 aberration C21, ~few hundred nanometers
aberration_coefficients = {'C10': defocus, 'C30': Cs, 'C12': astig, 'C21': coma}



#import preparation
data_dir = './data/data_DFT/'
data_names = [entry for entry in os.listdir(data_dir) if entry.endswith('.traj')]

#fill path_list with imported data_names
#for i in range(len(data_names)):
#    path_i = f'{data_dir}{data_names[i]}'
#    path_list[i] = path_i


#----------------------------------------------------------------------------------------------------

#simulation function

def simulation(path_or, path_res):

    #import data
    atoms = read(path_or ,index=':')   
    hBN_manipulated = atoms[-1]
        
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
    measurements_total.to_zarr(f'./data_DFT/results/{path_res}_RAW.zarr')

    #add to overview dataframe by appending new row with df.loc[index]
    #df = pd.read_csv('./data/data_overview.csv')
    #df.loc[len(df.index)] = [value for value in variables_list]
    #df.to_csv('./data/data_overview.csv', index=False)

#----------------------------------------------------------------------------------------------------

#run code

if __name__ == "__main__":
    count = 0
    try: 
    #run simulation for all variations
        for data in data_names:
            path_origin = (f'{data_dir}{data}')       
            timestamp = timeformat()
            path_result = (f'{data_dir}result/{timestamp}_{data}')
            print(path_result)
            count += 1
            simulation(path_origin, path_result)
            print(count)
                    
        
    except Exception as e:
        print(f"Oups - something went wrong: {e}")
