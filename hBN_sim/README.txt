PROJECT DESCRIPTION:
This projects was generated to simulate STEM-images of hBN structures with varying defect atoms.


SETUP:
a) via Conda:
>>>conda env create --file hBN_sim.yml
>>>conda activate hBN_sim

b) via Pip:
>>>python3.9 -m venv env
>>>source env/bin/activate
>>>pip install -r requirements.txt


PROJECT STRUCTURE
i) Testing
The testing folder contains the main jupyter notebook where snippets of the final simulation can be tested.
There is also the possibility of generating series of f.e. aberration versions.

ii) Simulations
The simulations themselves are written in .py files. To run, f.e:
>>> cd to simulations
>>> python hBN_simulation.py 

iii) Data
The generated data will be saved in the data folder(s) (.zarr files)

iv) Postprocessing
The jupyter notebooks for postprocessing allow to generate images and/or lineprofiles of the data.

v) Results
The postprocessed images (.png format) are being saved in the results folder.


FEATURES (simulation & postprocessing)
- no aberrations
- with aberrations
- convergence analysis of number of phonon configurations
- processing of DFT relaxed structure (.traj) (NOT generation of those structures)
- more detailed examination of differences between C and B defects
