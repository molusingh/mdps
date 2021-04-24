# cs7641-mdp

# Conda Setup Instructions (need conda, can get miniconda off chocolatey for windows or homebrew on mac)
### Using conda to create python environment
conda env create -f environment.yml

### activate the environemnt
conda activate cs7641

### if needed, add debugger
jupyter labextension install @jupyterlab/debugger

### add mdptoolbox-hiive
pip install mdptoolbox-hiive

### update environment after changes to environment.yml file (deactivate env first)
conda env update --file environment.yml --prune

### Open up jupyter lab to access notebook if desired
jupyter lab

# generate final results
python main.py 

References:
MDP Toolbox library 
https://github.com/hiive/hiivemdptoolbox
Piazza
