#shebang?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abtem
import ase
import dask
import os 
import sys

#paths
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_dir = os.path.abspath(os.path.join(parent_dir, 'data'))

#import homemade modules by adding hBN_sim directory to sys.path
sys.path.insert(0, parent_dir)
from helper_funcs import timeformat, built_structure, defect_indices, structure_manipulation

#Parameters & Settings
#----------------------------------------------------------------------------------------------------
#configuration
abtem.config.set({"device": "gpu", "fft": "fftw"})
dask.config.set({"num_workers": 1})

#FIX
lattice_constant = 2.504 # https://www.sciencedirect.com/science/article/abs/pii/S0025540815300088
size_x = 20 #size of structure
size_y = 13
energy_probe = 60e3
defocus = 0  # 0 or f.e.: 'scherzer'
semiangle_cutoff = 30 
Cs = -10e-6*1e10 #spherical aberration; 10 micrometers

#VARIABLE
vacancy_atom_list = ['B', 'N'] # 'N' for replacement of N-atom, or 'B' for replacement of B-atom
defect_atom_symbol_list = ['O', 'C', 'N', 'B', 'Si'] # symbols of impurity atoms for single defects
double_defects_list = ['O', 'C', 'Si']# symbols of impurity atoms for double defects

#abberrations: [scale: Cs = -10e-6*1e10; astig = 10; coma = 1000]
astig = 0 #f.e. [0, 15, 25] aberration C12, ~few nanometer, input in Å
astig_angle = 0 #f.e. [-0.3, 0, 0.3] #rad
coma = 0 #f.e. [0, 1000, 1500] aberration C21, ~few hundred nanometers, input in Å
aberration_coefficients = {'C10': defocus, 'C30': Cs, 'C12': astig, 'C21': coma}


#----------------------------------------------------------------------------------------------------

#simulation function

def simulation(vacancy_atom, defect_atom_symbol, defect_type, path, variables_list):
    
    #build structure
    hBN = built_structure(size_x, size_y)
    #get defect indices
    ind_array = defect_indices(hBN, vacancy_atom) #returns array of form: [ind_single, ind_double, ind_ring]
    #insert impurities
    hBN_manipulated = structure_manipulation(hBN, defect_type, defect_atom_symbol, ind_array)
        
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
    measurements_total.to_zarr(f'{data_dir}/{path}_RAW.zarr')
    
    #add to overview dataframe by appending new row with df.loc[index]
    df = pd.read_csv(f'{data_dir}/data_overview.csv')
    df.loc[len(df.index)] = [value for value in variables_list]
    df.to_csv(f'{data_dir}/data_overview.csv', index=False)

    
#----------------------------------------------------------------------------------------------------

#run code

if __name__ == "__main__":
    count = 0
    try: 
        #run simulation for all variations
        
        #SINGLE DEFECTS
        for atom in vacancy_atom_list:
            for symbol in defect_atom_symbol_list:
                d_type = 'single'
                
                #check if vacany atom and filling defect atom are the same, else run simulation
                if atom == symbol:
                    pass
                    
                else:
                    #generate unique filename with parameter info
                    timestamp = timeformat()
                    path = (f'{timestamp}_hBN_size{size_x}x{size_y}_{d_type}_vacancy{atom}'\
                             f'_filledwith{symbol}_energy{int(energy_probe)}_defocus{defocus}'\
                             f'_Cs{int(Cs)}_astig{astig}_astigangle{astig_angle}_coma{coma}')
                    print(path)
                    
                    #variables_list is only needed for data_overview.csv file
                    variables_list = [timestamp, str(size_x) + 'x' + str(size_y), d_type, atom, symbol, 
                                      int(energy_probe), defocus, semiangle_cutoff, int(Cs), astig,
                                      astig_angle, coma]                    
                    #run sim
                    simulation(atom, symbol, d_type, path, variables_list)
                    count += 1
                    print(count)
                    
        #DOUBLE DEFECTS
        for symbol in double_defects_list:
            d_type = 'double'
            atom = 'B' #setting vacancy atom for sim -> doesn't matter which one for outcome

            #generate unique filename with parameter info
            timestamp = timeformat()
            path = (f'{timestamp}_hBN_size{size_x}x{size_y}_{d_type}_vacancy{atom}'\
                    f'_filledwith{symbol}_energy{int(energy_probe)}_defocus{defocus}'\
                    f'_Cs{int(Cs)}_astig{astig}_astigangle{astig_angle}_coma{coma}')
            print(path)
            
            #variables_list is only needed for data_overview.csv file
            variables_list = [timestamp, str(size_x) + 'x' + str(size_y), d_type, atom, 
                              symbol, int(energy_probe), defocus, semiangle_cutoff, int(Cs), 
                              astig, astig_angle, coma]
            
            #run sim
            simulation(atom, symbol, d_type, path, variables_list)
            count += 1
            print(count)
        
    except Exception as e:
        print(f"Oups - something went wrong: {e}")
