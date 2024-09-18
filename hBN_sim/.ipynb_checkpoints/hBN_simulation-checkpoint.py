#shebang?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abtem
import ase
import dask

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

#VARIABLE
vacancy_atom_list = ['B', 'N'] # 'N' for replacement of N-atom, or 'B' for replacement of B-atom
defect_atom_symbol_list = ['O', 'C', 'N', 'B', 'Si'] # symbols of impurity atoms
#defect_type_list = ['single', 'double'] #-> simulation is programmed to generate single & double defect images -> see below
double_defects_list = ['O', 'C', 'Si']

#abberrations: [scale: Cs = -10e-6*1e10; astig = 10; coma = 1000]
astig = 0 #[0, 15, 25] #25 aberration C12, ~few nanometers
astig_angle = 0 #[-0.3, 0, 0.3] #+/-0.3 #rad
coma = 0 #[0, 1000, 1500] #1000 aberration C21, ~few hundred nanometers
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
    measurements_total.to_zarr(f'./data/{path}_RAW.zarr')

    #add to overview dataframe by appending new row with df.loc[index]
    df = pd.read_csv('./data/data_overview.csv')
    df.loc[len(df.index)] = [value for value in variables_list]
    df.to_csv('./data/data_overview.csv', index=False)

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
                    timestamp = timeformat()
                    path = (f'{timestamp}_hBN_size{size_x}x{size_y}_{d_type}_vacancy{atom}_filledwith{symbol}_energy{int(energy_probe)}_defocus{defocus}_Cs{int(Cs)}_astig{astig}_astigangle{astig_angle}_coma{coma}')#_focal{int(focal_spread)}_ang{int(angular_spread)}')
                    variables_list = [timestamp, str(size_x) + 'x' + str(size_y), d_type, atom, symbol, int(energy_probe), 
                                      defocus, semiangle_cutoff, int(Cs), astig, astig_angle, coma]
                    print(path)
                    count += 1
                    simulation(atom, symbol, d_type, path, variables_list)
                    print(count)
                    
        #DOUBLE DEFECTS
        for symbol in double_defects_list:
            d_type = 'double'
            atom = 'B' #setting vacancy atom for sim -> doesn't matter which one for outcome
            timestamp = timeformat()
            path = (f'{timestamp}_hBN_size{size_x}x{size_y}_{d_type}_vacancy{atom}_filledwith{symbol}_energy{int(energy_probe)}_defocus{defocus}_Cs{int(Cs)}_astig{astig}_astigangle{astig_angle}_coma{coma}')#_focal{int(focal_spread)}_ang{int(angular_spread)}')
            variables_list = [timestamp, str(size_x) + 'x' + str(size_y), d_type, atom, symbol, int(energy_probe), 
                              defocus, semiangle_cutoff, int(Cs), astig, astig_angle, coma]
            print(path)
            count += 1
            simulation(atom, symbol, d_type, path, variables_list)
            print(count)
        
    except Exception as e:
        print(f"Oups - something went wrong: {e}")
