import ase
import abtem
import numpy as np
from datetime import datetime

def timeformat():
    '''creates timestamp of format YYYYMMDDHHMMSS'''
    #get current time
    now = datetime.now()
    now_str = [str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second),]
    
    #make sure date has same format every time: YYYYMMDDHHMMSS
    for i in range(len(now_str)):
        if len(now_str[i]) == 1:
            now_str[i] = '0' + now_str[i]
    now_str = ''.join(now_str)
    
    return(now_str)

def test():
    return('test')

def built_structure(size_x, size_y, lattice_constant=2.504):
    '''creation of orthogonalized hBN structure; positional arguments: size_x, size_y, returns: atomic structure'''
    #create hBN cell
    hBN_cell = ase.build.graphene(formula='BN', a=lattice_constant, thickness=0.0, size=(1, 1, 1), vacuum=2)
    #orthogonalize cell and create structure
    hBN_orth = abtem.orthogonalize_cell(hBN_cell)
    hBN = hBN_orth*(size_x, size_y, 1)
    return(hBN)

def defect_indices(hBN, vacancy_atom, lattice_constant=2.504):
    '''finds center indices of atomic structure; positional argument: atomic structure; returns: array of indices'''
    #find atoms located in the center
    x_center = max(hBN.positions[:, 0])/2
    y_center = max(hBN.positions[:, 1])/2
    mask_center_position_x = ((hBN.positions[:,0] < x_center + lattice_constant)*
                              (hBN.positions[:,0] > x_center - lattice_constant))
    mask_center_position_y = ((hBN.positions[:,1] < y_center + lattice_constant)*
                              (hBN.positions[:,1] > y_center - lattice_constant))
    mask_center = mask_center_position_x * mask_center_position_y
    center_indices = (np.asarray(np.where(mask_center == True)))[0]

    #SINGLE: find index values in the middle of center index array to adress one (or two) specific atom(s)
    ind = int(len(center_indices)/2 - 1) #center_indices.shape = (1,8);
    ind_random = center_indices[ind]
    #check if chosen atom is wanted single defect type and/or (re-)assign
    if hBN.symbols[ind_random] == vacancy_atom:
        ind_single = ind_random
    else:
        ind_single = center_indices[ind + 2]
        # +2 because Atoms are saved in the structure [... N N B B N N B...]
    
    #for DOUBLE: make sure, the two atoms are neighbours
    #find distance of next neighbours
    nnd = round(hBN.get_distance(1, 3),4)
    
    for index in center_indices:
        index_dist = round(hBN.get_distance(ind_single, index),4)
        if index_dist == nnd:
            #found next neighbour -> assign second index
            ind_double = [ind_single, index]
            break
        else:
            pass
    
    #for RING:
    #find distances of next neighbours, second neighbours (nnd_2) and third neighbors (nnd_3)
    nnd = round(hBN.get_distance(1, 3),4)
    nnd_2 = round(hBN.get_distance(1, 2),4)
    nnd_3 = round(hBN.get_distance(0,2),4)#dist for atoms opposed to each other
    
    #create dummy array for ring indices and assign first two possitions with double indices
    ind_ring = [0 for i in range(6)]
    ind_ring[0], ind_ring[1] = ind_double
    
    #find atom opposite to first atom
    for index in center_indices:
        dist_to_1 = round(hBN.get_distance(ind_ring[0], index),4)
        dist_to_2 = round(hBN.get_distance(ind_ring[1], index),4)
    
        if dist_to_1 == nnd_3 and dist_to_2 == nnd_2:
            ind_ring[3] = index #position 4
            break
        else:
            pass
    
    #find remaining atoms (position numbering 1 to 6 counterclockwise): 
    for index in center_indices:
        dist_to_1 = round(hBN.get_distance(ind_ring[0], index),4)
        dist_to_2 = round(hBN.get_distance(ind_ring[1], index),4)
        dist_to_4 = round(hBN.get_distance(ind_ring[3], index),4)
        if dist_to_1 == nnd_2 and dist_to_2 == nnd and dist_to_4 == nnd:
            ind_ring[2] = index #position 3
        if dist_to_1 == nnd_2 and dist_to_2 == nnd_3 and dist_to_4 == nnd:
            ind_ring[4] = index #position 5
        if dist_to_1 == nnd and dist_to_2 == nnd_2 and dist_to_4 == nnd_2:
            ind_ring[5] = index #position 6 
        else:
            pass
    
    return(ind_single, ind_double, ind_ring)

def structure_manipulation(structure, defect_type, defect_atom_symbol, ind_array):
    ind_single, ind_double, ind_ring = ind_array
    #insert impurity atoms - introducing single and double defects
    hBN_manipulated = structure.copy()

    if defect_type == 'single':
        hBN_manipulated.symbols[ind_single] = defect_atom_symbol
    elif defect_type == 'double':
        hBN_manipulated.symbols[ind_double] = defect_atom_symbol
    elif defect_type == 'ring':
        hBN_manipulated.symbols[ind_ring] = defect_atom_symbol
    else: 
        raise

    return(hBN_manipulated)
    
   
def nnd():
    hBN = built_structure(4,4)
    nnd = round(hBN.get_distance(1, 3),4)
    return(nnd)