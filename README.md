# README

Code Structure:

* src
    * bbDebiasing.py: Implementation of "bias bounties" method of debiasing
    * maxEnsembleDebias.py: Implementation of max ensembling method of debiasing. 
    * policies.py: Policies. Currently just takes max coordinate. 
* experiments
    * synthetic-experiments.ipynb: Synthetic Experiments
    * electricity-experiments.ipynb: Initial experiments on electric transformer dataset
    * options-experiments.ipynb: Currently just loading in the data, haven't written any experiments. 
* old-work
    * A bunch of garbage code that I don't want to get rid of. 
* testing
    * Various testing of code functionality

To build environment:
conda env create -f environment.yml
conda activate bb-portfolio


Because of problems with the default opensource optimization solvers used by cvxpy, have switched to Gurobi. This requires a separate download of Gurobi and a license; they have academic licenses for free. 

See https://www.gurobi.com/features/academic-named-user-license/ for instructions. After downloading, if running code in VSCode, will need to quit and reopen VSCode. 

